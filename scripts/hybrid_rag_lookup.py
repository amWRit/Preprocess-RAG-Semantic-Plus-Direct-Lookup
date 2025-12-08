import os
import json
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
import boto3
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.local'))

# Initialize AWS Bedrock client and embeddings
AWS_REGION = os.getenv('AWS_REGION')
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
)
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

# Load FAISS index
faiss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public', 'vector-store'))
faiss_index_path = os.path.join(faiss_dir, 'index.faiss')
faiss_pkl_path = os.path.join(faiss_dir, 'index.pkl')

if not (os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path)):
    print(f"[-] FAISS index files not found at {faiss_dir}")
    print("[!] Please run 'python scripts/preprocess_docs.py --rebuild' to create the index.")
    exit(1)

db = FAISS.load_local(
    folder_path=faiss_dir,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

def semantic_lookup(query, top_k=3):
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context

def generate_answer(context, question):
    # Format messages for Bedrock
    messages = [
        {
            "role": "user",
            "content": [
                {"text": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        }
    ]
    body = {
        "messages": messages,
        "system": [
            {"text": "You are a helpful assistant. Use the provided context to answer the question. If you don't know, say so."}
        ]
    }
    response = bedrock_client.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    response_body = json.loads(response['body'].read())
    return response_body['output']['message']['content'][0]['text']

if __name__ == "__main__":
    question = "Who are the staff members?"
    context = semantic_lookup(question)
    print("Retrieved context:\n", context)
    answer = generate_answer(context, question)
    print("\nGenerated answer:\n", answer)