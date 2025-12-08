# !pip install -qU langchain-aws langchain langchain-classic langgraph langchain-chroma langchain-community pypdf PyPDF2 langchain-text-splitters langchain-google-community unstructured[pdf] langchain-confluence google-api-python-client google-auth-httplib2 google-auth-oauthlib atlassian-python-api rich

import os
import json
import boto3
from google.colab import userdata
AWS_ACCESS_KEY_ID = userdata.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = userdata.get('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = userdata.get('AWS_SESSION_TOKEN')
AWS_REGION = userdata.get('AWS_REGION')

bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
        )

from langchain_aws import ChatBedrock

llm_provider = "bedrock"
def get_llm(model=None, temperature=0):
  if llm_provider == "bedrock":
        return ChatBedrock(
            client=bedrock_client,
            model_id=model or "amazon.nova-lite-v1:0",
            temperature=temperature
        )

model = get_llm()
llm = get_llm()

from langchain_text_splitters  import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

from langchain_aws import BedrockEmbeddings
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")

all_chunks = []

# Load PDF
from langchain_community.document_loaders import PyPDFLoader, ConfluenceLoader

pdf_file_path = "./data/LF-Policy.pdf"
loader_pdf = PyPDFLoader(pdf_file_path)
documents_pdf = loader_pdf.load()

chunks_pdf = text_splitter.split_documents(documents_pdf)
all_chunks.extend(chunks_pdf)

import pprint
pprint.pp(chunks_pdf[0].metadata)


#Vector DB
from langchain_chroma import Chroma as ch

persist_directory = "chroma_db"

db = ch.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

stored_data = db.get()

print("Stored IDs:", stored_data['ids'])
print("Stored Metadatas:", stored_data['metadatas'])
print("Stored Documents:", stored_data['documents'])

# Retrieve
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

lcel = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Generate
# Example
query = "What are the core values of Leapfrog as stated in the handbook?"
response = lcel.invoke(query)
print(response)


from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

# Example

lf_policy_queries = [
    "How do I report harassment at Leapfrog?",
    "What constitutes sexual harassment according to the policy?",
    "How long does the harassment investigation process take?",
    "Will I face retaliation for reporting harassment?",
    "Who investigates harassment complaints and what is their role?",
    "What are the different types of harassment covered in this policy?",
    "What should I do if my manager or supervisor is harassing me?",
    "What disciplinary actions can be taken against someone found guilty of harassment?",
    "Can harassment cases be resolved through conciliation?",
    "What support is available for victims of harassment?"
]

lf_nomination_queries = [
    "What are the criteria for 'Exceeds Expectations' performance rating?",
    "What defines a 'Below Expectations' employee in the appraisal process?",
    "How is technical expertise evaluated in the performance appraisal?",
    "What are the key indicators of strong ownership in performance reviews?",
    "How is communication assessed during employee appraisals?",
    "What does it mean to be a good team player according to the nomination guidelines?",
    "What happens if an employee consistently fails to meet deadlines?",
    "How are leadership skills evaluated in the performance appraisal?",
    "What is the difference between 'Exceeds Expectations' and 'Below Expectations' in punctuality?",
    "How does active participation in company initiatives impact performance ratings?"
]

lf_code_of_conduct_queries = [
    "What are Leapfrog's core values and how do they guide employee behavior?",
    "How do I report ethical violations or misconduct at Leapfrog?",
    "What constitutes a conflict of interest and how should I disclose it?",
    "What is Leapfrog's policy on accepting gifts from vendors or clients?",
    "How does Leapfrog protect confidential information and intellectual property?",
    "What are the consequences of violating the Code of Conduct?",
    "What is Leapfrog's stance on bribery, corruption, and anti-money laundering?",
    "How should I handle situations where a customer uses offensive language?",
    "What are my responsibilities regarding company assets and expense claims?",
    "What protection do I have if I report misconduct through speak-up channels?"
]

# queries = lf_policy_queries + lf_nomination_queries + lf_nomination_queries

# print("##########################################################")
# print("Queries and their response related to Anti-Harassment Policy")
# print("##########################################################\n")
# for query in lf_policy_queries:
#   print(query)
#   response = lcel.invoke(query)
#   print(response)
#   print("\n")

console = Console()



