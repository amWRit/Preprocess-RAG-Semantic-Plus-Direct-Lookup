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

# 1. Define Pydantic models for different document types

# Staff model
class StaffMember(BaseModel):
    name: str = Field(description="name of the staff member")
    role: str = Field(description="role or designation of the staff member")
    bio: str = Field(description="short biography or description of the staff member")

class StaffData(BaseModel):
    staff_members: list[StaffMember] = Field(description="List of staff members with their name, role, and bio")

# Contact model
class Contact(BaseModel):
    name: str = Field(description="name of the contact person")
    email: str = Field(description="email address")
    phone: str = Field(description="phone number")
    organization: str = Field(description="organization or company name")

class ContactData(BaseModel):
    contacts: list[Contact] = Field(description="List of contacts with their details")

# Partner model
class Partner(BaseModel):
    name: str = Field(description="name of the partner organization")
    type: str = Field(description="type of partnership (e.g., technology partner, strategic partner)")
    description: str = Field(description="description of the partnership")
    contact_person: str = Field(description="main contact person at the partner organization")

class PartnerData(BaseModel):
    partners: list[Partner] = Field(description="List of partner organizations with their details")

# Generic configuration for each document type
DOCUMENT_TYPES_CONFIG = {
    "staff.pdf": {
        "pydantic_model": StaffData,
        "extraction_field": "staff_members",
        "system_prompt": "Extract information about staff members from the provided text. Ensure the output is a JSON list of staff members following the schema below:\n{format_instructions}",
        "metadata_fields": ["name", "role"]
    },
    "contacts.pdf": {
        "pydantic_model": ContactData,
        "extraction_field": "contacts",
        "system_prompt": "Extract contact information from the provided text. Ensure the output is a JSON list of contacts following the schema below:\n{format_instructions}",
        "metadata_fields": ["name", "email"]
    },
    "partners-and-supporters.pdf": {
        "pydantic_model": PartnerData,
        "extraction_field": "partners",
        "system_prompt": "Extract partner information from the provided text. Ensure the output is a JSON list of partners following the schema below:\n{format_instructions}",
        "metadata_fields": ["name", "type"]
    }
}

# 1. Create directories and move files
unstructured_data_dir = "./data/unstructured"
structured_data_dir = "./data/structured"

# Create directories if they don't exist
os.makedirs(unstructured_data_dir, exist_ok=True)
os.makedirs(structured_data_dir, exist_ok=True)

# 2. Check if Chroma DB already exists
unified_persist_directory = "unified_chroma_db"
db_exists = os.path.exists(unified_persist_directory)

# 3. Initialize embeddings
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")

# 4. If DB doesn't exist, process all documents and create it
if not db_exists:
    print("\n🔄 Chroma DB not found. Processing documents...")
    
    # Initialize all_chunks and text_splitter
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Process unstructured data
    print("\n📄 Processing unstructured data...")
    for filename in os.listdir(unstructured_data_dir):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(unstructured_data_dir, filename)
            loader_pdf = PyPDFLoader(pdf_file_path)
            documents_pdf = loader_pdf.load()
            chunks_pdf = text_splitter.split_documents(documents_pdf)
            all_chunks.extend(chunks_pdf)
            print(f"Loaded and chunked {len(documents_pdf)} pages from {pdf_file_path}, resulting in {len(chunks_pdf)} chunks.")

    # Process structured data
    print("\n📋 Processing structured data...")
    all_structured_data_list = []
    all_extracted_items = []

    for filename in os.listdir(structured_data_dir):
        if filename.endswith(".pdf"):
            # Check if this file type has a configuration
            if filename not in DOCUMENT_TYPES_CONFIG:
                print(f"⚠️ No configuration found for {filename}. Skipping...")
                continue
            
            config = DOCUMENT_TYPES_CONFIG[filename]
            pydantic_model = config["pydantic_model"]
            extraction_field = config["extraction_field"]
            system_prompt = config["system_prompt"]
            metadata_fields = config["metadata_fields"]
            
            pdf_file_path = os.path.join(structured_data_dir, filename)
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()

            raw_text = ""
            for doc in documents:
                raw_text += doc.page_content

            print(f"Extracted raw text from {pdf_file_path}. Length: {len(raw_text)}")

            # Create the prompt template for structured extraction
            structured_parser = PydanticOutputParser(pydantic_object=pydantic_model)
            
            structured_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
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

            # Invoke the chain to extract structured data
            try:
                extracted_data = structured_extraction_chain.invoke(raw_text)
                items = getattr(extracted_data, extraction_field)
                all_structured_data_list.extend(items)
                all_extracted_items.append({
                    "filename": filename,
                    "field_name": extraction_field,
                    "count": len(items),
                    "items": items
                })
                print(f"✓ Extracted {len(items)} items from {filename}.")
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")

    # 5. Save combined structured data to JSON
    output_structured_file_path = "all_structured_data.json"
    with open(output_structured_file_path, "w") as f:
        # Create a dictionary with all extracted data organized by document type
        output_data = {}
        for item_group in all_extracted_items:
            field_name = item_group["field_name"]
            items = item_group["items"]
            output_data[field_name] = [item.model_dump() for item in items]
        
        json.dump(output_data, f, indent=2)
    print(f"Combined structured data saved to {output_structured_file_path}")

    # 6. Convert structured data to Document objects
    documents_from_structured = []
    for item_group in all_extracted_items:
        filename = item_group["filename"]
        field_name = item_group["field_name"]
        items = item_group["items"]
        
        for item in items:
            item_dict = item.model_dump()
            # Create content from all fields
            content = "\n".join([f"{key}: {value}" for key, value in item_dict.items()])
            
            # Create metadata with the configured fields
            config = DOCUMENT_TYPES_CONFIG[filename]
            metadata_fields = config["metadata_fields"]
            metadata = {"source": filename, "type": field_name}
            for field in metadata_fields:
                if field in item_dict:
                    metadata[field] = str(item_dict[field])
            
            documents_from_structured.append(Document(page_content=content, metadata=metadata))

    print(f"Converted {len(documents_from_structured)} structured entries into Document objects.")

    # 7. Extend all_chunks with documents from structured data
    all_chunks.extend(documents_from_structured)
    print(f"Total documents for RAG (unstructured + structured): {len(all_chunks)}")

    # 8. Create unified Chroma DB
    print(f"Creating new Chroma DB at {unified_persist_directory}...")
    db_unified = ch.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=unified_persist_directory
    )
    print(f"✓ Chroma DB created at {unified_persist_directory} with {db_unified._collection.count()} documents.")
else:
    # DB already exists, just load it
    print(f"\n✓ Loading existing Chroma DB from {unified_persist_directory}...")
    db_unified = ch(
        persist_directory=unified_persist_directory,
        embedding_function=embeddings
    )
    print(f"✓ Loaded existing Chroma DB with {db_unified._collection.count()} documents.")

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