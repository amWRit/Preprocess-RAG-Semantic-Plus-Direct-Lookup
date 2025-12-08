# !pip install -qU langchain-aws langchain langchain-classic langgraph langchain-chroma langchain-community pypdf PyPDF2 langchain-text-splitters langchain-google-community unstructured[pdf] rich

import os
import json
import pprint
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma as ch
from langchain_classic import hub
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Load environment variables from .env file
load_dotenv()

console = Console()

# 0. Initialize AWS Bedrock client
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
AWS_REGION = os.getenv('AWS_REGION')

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="amazon.nova-lite-v1:0",
    temperature=0
)

# 1. Define Pydantic models
class StaffMember(BaseModel):
    name: str = Field(description="name of the staff member")
    role: str = Field(description="role or designation of the staff member")
    bio: str = Field(description="short biography or description of the staff member")

class StaffData(BaseModel):
    staff_members: list[StaffMember] = Field(description="List of staff members with their name, role, and bio")

# 1. Create directories and move files
unstructured_data_dir = "./data/unstructured"
structured_data_dir = "./data/structured"

# Create directories if they don't exist
os.makedirs(unstructured_data_dir, exist_ok=True)
os.makedirs(structured_data_dir, exist_ok=True)

# 2. Initialize all_chunks, text_splitter, and embeddings
all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")

# 3. Process unstructured data (LF-Policy.pdf)
print("\nProcessing unstructured data...")
for filename in os.listdir(unstructured_data_dir):
    if filename.endswith(".pdf"):
        pdf_file_path = os.path.join(unstructured_data_dir, filename)
        loader_pdf = PyPDFLoader(pdf_file_path)
        documents_pdf = loader_pdf.load()
        chunks_pdf = text_splitter.split_documents(documents_pdf)
        all_chunks.extend(chunks_pdf)
        print(f"Loaded and chunked {len(documents_pdf)} pages from {pdf_file_path}, resulting in {len(chunks_pdf)} chunks.")

# 4. Process structured data (staff.pdf)
print("\nProcessing structured data...")
all_structured_data_list = []
structured_parser = PydanticOutputParser(pydantic_object=StaffData)

for filename in os.listdir(structured_data_dir):
    if filename.endswith(".pdf"):
        staff_pdf_file_path = os.path.join(structured_data_dir, filename)
        loader_staff = PyPDFLoader(staff_pdf_file_path)
        documents_staff = loader_staff.load()

        raw_staff_text = ""
        for doc in documents_staff:
            raw_staff_text += doc.page_content

        print(f"Extracted raw text from {staff_pdf_file_path}. Length: {len(raw_staff_text)}")

        # Create the prompt template for structured extraction
        structured_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Extract information about staff members from the provided text. Ensure the output is a JSON list of dictionaries, following the schema below:\n{format_instructions}"),
                ("human", "Text: {text}")
            ]
        ).partial(format_instructions=structured_parser.get_format_instructions())

        # Construct the LLM chain for structured extraction
        structured_extraction_chain = (
            {"text": RunnablePassthrough()}
            | structured_prompt
            | llm
            | structured_parser
        )

        # Invoke the chain to extract structured staff data
        structured_staff_data = structured_extraction_chain.invoke(raw_staff_text)
        all_structured_data_list.extend(structured_staff_data.staff_members)
        print(f"Extracted {len(structured_staff_data.staff_members)} staff members from {staff_pdf_file_path}.")

# 5. Save combined structured data to JSON
output_structured_file_path = "all_structured_data.json"
with open(output_structured_file_path, "w") as f:
    json.dump({'staff_members': [member.model_dump() for member in all_structured_data_list]}, f, indent=2)
print(f"Combined structured staff data saved to {output_structured_file_path}")

# 6. Convert structured data to Document objects
staff_documents = []
for staff_member_dict in [member.model_dump() for member in all_structured_data_list]:
    content = f"Name: {staff_member_dict['name']}\nRole: {staff_member_dict['role']}\nBio: {staff_member_dict['bio']}"
    metadata = {"source": "staff.json", "name": staff_member_dict['name'], "role": staff_member_dict['role']}
    staff_documents.append(Document(page_content=content, metadata=metadata))
print(f"Converted {len(staff_documents)} structured staff entries into Document objects.")

# 7. Extend all_chunks with staff_documents
all_chunks.extend(staff_documents)
print(f"Total documents for RAG (unstructured + structured): {len(all_chunks)}")

# 8. Create unified Chroma DB
unified_persist_directory = "unified_chroma_db"
db_unified = ch.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=unified_persist_directory
)
print(f"Unified Chroma DB created at {unified_persist_directory} with {db_unified._collection.count()} documents.")

# 9. Re-initialize RAG chain
retriever_unified = db_unified.as_retriever()
rag_prompt = hub.pull("rlm/rag-prompt")

lcel_unified = (
    {"context": retriever_unified, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("RAG chain re-initialized with the unified Chroma DB.")

# 10. Define test queries
lf_policy_queries = [
    "How do I report harassment at Leapfrog?",
    # "What constitutes sexual harassment according to the policy?",
    # "How long does the harassment investigation process take?",
    # "Will I face retaliation for reporting harassment?",
    # "Who investigates harassment complaints and what is their role?",
    # "What are the different types of harassment covered in this policy?",
    # "What should I do if my manager or supervisor is harassing me?",
    # "What disciplinary actions can be taken against someone found guilty of harassment?",
    # "Can harassment cases be resolved through conciliation?",
    # "What support is available for victims of harassment?"
]

lf_staff_queries = [
    "Who is Swastika Shrestha and what is her role?",
    # "Tell me about Krishna Kumar K.C.",
    # "What is Khika Prasad Nepal's role and bio?",
    # "Who is Anjali Shrestha and what is her designation?",
    # "What is Manish Shrestha's role?",
    # "Can you tell me about the Team Leader of the Fellowship team?",
    # "Who is the Finance Manager?",
    # "What is Sirjana Dhital's role?"
]

# 11. Run test queries and print results
console.print(Panel.fit(
    Text("🛡️ Anti-Harassment Policy Queries (Unified RAG)", style="bold cyan", justify="center"),
    border_style="bright_blue", padding=(1, 2)
))

for i, query in enumerate(lf_policy_queries, 1):
    console.print(f"\n[bold yellow]Q{i}:[/bold yellow] {query}")
    with console.status("[bold green]Generating response..."):
        response = lcel_unified.invoke(query)
    console.print(Panel(
        Text(response, style="#36312F"),
        title=f"Response {i}",
        border_style="green",
        padding=(1, 2),
        expand=False
    ))
    console.print("[dim]─" * 80)

console.print(Panel.fit(
    Text("🧑‍💻 Staff Information Queries (Unified RAG)", style="bold magenta", justify="center"),
    border_style="bright_magenta", padding=(1, 2)
))

for i, query in enumerate(lf_staff_queries, 1):
    console.print(f"\n[bold yellow]Q{i}:[/bold yellow] {query}")
    with console.status("[bold green]Generating response..."):
        response = lcel_unified.invoke(query)
    console.print(Panel(
        Text(response, style="#36312F"),
        title=f"Response {i}",
        border_style="green",
        padding=(1, 2),
        expand=False
    ))
    console.print("[dim]─" * 80)