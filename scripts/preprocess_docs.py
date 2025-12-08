#!/usr/bin/env python3
"""
TFN-AI RAG Preprocessing Script with Pre-computed Embeddings (FAISS Version)
Processes PDFs, generates embeddings, and saves vector store for instant loading

Usage:
    python scripts/preprocess_docs.py

Requirements:
    pip install langchain-text-splitters langchain-community langchain-aws boto3 faiss-cpu python-dotenv
"""

import os
import json
import argparse
import shutil
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# ==================== Pydantic Models ====================

class StaffMember(BaseModel):
    name: str = Field(description="name of the staff member")
    role: str = Field(description="role or designation of the staff member")
    bio: str = Field(description="short biography or description of the staff member")

class StaffData(BaseModel):
    staff_members: list[StaffMember] = Field(description="List of staff members with their name, role, and bio")

class Contact(BaseModel):
    name: str = Field(description="name of the contact person")
    email: str = Field(description="email address")
    phone: str = Field(description="phone number")
    organization: str = Field(description="organization or company name")

class ContactData(BaseModel):
    contacts: list[Contact] = Field(description="List of contacts with their details")

class Partner(BaseModel):
    name: str = Field(description="name of the partner organization")
    type: str = Field(description="type of partnership (e.g., technology partner, strategic partner)")
    description: str = Field(description="description of the partnership")
    contact_person: str = Field(description="main contact person at the partner organization")

class PartnerData(BaseModel):
    partners: list[Partner] = Field(description="List of partner organizations with their details")


# ==================== Configuration & Constants ====================

# AWS Bedrock Models
BEDROCK_LLM_MODEL_ID = "amazon.nova-lite-v1:0"
BEDROCK_EMBEDDINGS_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File Extensions
PDF_EXTENSION = ".pdf"

# Document Type Configuration
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


# ==================== DocumentPreprocessor Class ====================

class DocumentPreprocessor:
    """
    A modular class for preprocessing documents and building FAISS vector store.
    """
    
    def __init__(self, 
                 unstructured_dir=None,
                 structured_dir=None,
                 faiss_dir=None,
                 output_json=None):
        """
        Initialize the preprocessor with directory paths.
        Paths are relative to the project root (parent of scripts folder).
        """
        # Get project root directory (parent of scripts folder)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default paths relative to project root
        self.unstructured_dir = unstructured_dir or os.path.join(project_root, "data", "unstructured")
        self.structured_dir = structured_dir or os.path.join(project_root, "data", "structured")
        self.faiss_dir = faiss_dir or os.path.join(project_root, "public", "vector-store")
        self.output_json = output_json or os.path.join(project_root, "public", "all_structured_data.json")
        self.faiss_index_path = os.path.join(self.faiss_dir, "index.faiss")
        
        # Initialize AWS and LLM
        self._init_aws_and_llm()
        
        # Initialize embeddings
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id=BEDROCK_EMBEDDINGS_MODEL_ID
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Create directories
        self._create_directories()
    
    def _init_aws_and_llm(self):
        """Initialize AWS Bedrock client and LLM."""
        # Load .env.local from parent directory
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
        load_dotenv(env_path)
        
        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
        AWS_REGION = os.getenv('AWS_REGION')
        
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
        )
        
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id=BEDROCK_LLM_MODEL_ID,
            temperature=0
        )
    
    def _create_directories(self):
        """Create required directories."""
        os.makedirs(self.unstructured_dir, exist_ok=True)
        os.makedirs(self.structured_dir, exist_ok=True)
    
    def _is_index_exists(self):
        """Check if FAISS index already exists."""
        return os.path.exists(self.faiss_index_path)
    
    def process_unstructured_data(self):
        """Process unstructured PDF documents and return chunks."""
        print("\n[*] Processing unstructured data...")
        all_chunks = []
        
        for filename in os.listdir(self.unstructured_dir):
            if filename.endswith(PDF_EXTENSION):
                pdf_path = os.path.join(self.unstructured_dir, filename)
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    chunks = self.text_splitter.split_documents(documents)
                    all_chunks.extend(chunks)
                    print(f"    [+] {filename}: {len(documents)} pages -> {len(chunks)} chunks")
                except Exception as e:
                    print(f"    [-] Error processing {filename}: {str(e)}")
        
        print(f"[+] Total unstructured chunks: {len(all_chunks)}")
        return all_chunks
    
    def _extract_structured_from_file(self, filename):
        """Extract structured data from a single PDF file."""
        if filename not in DOCUMENT_TYPES_CONFIG:
            print(f"    [!] No configuration for {filename}. Skipping...")
            return {}
        
        config = DOCUMENT_TYPES_CONFIG[filename]
        pdf_path = os.path.join(self.structured_dir, filename)
        
        try:
            # Load PDF and extract text
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            raw_text = "\n".join([doc.page_content for doc in documents])
            
            print(f"    [*] Processing {filename} ({len(raw_text)} chars)")
            
            # Setup extraction chain
            pydantic_model = config["pydantic_model"]
            parser = PydanticOutputParser(pydantic_object=pydantic_model)
            format_instructions = parser.get_format_instructions()
            
            # Create the system prompt with format instructions
            system_prompt_text = config["system_prompt"].format(format_instructions=format_instructions)
            
            # Build the user message with the text to extract from
            user_message = f"Text: {raw_text}"
            
            # Call Bedrock directly with proper message format
            import json
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": user_message
                        }
                    ]
                }
            ]
            
            # Create the body with proper format for nova-lite model
            body = {
                "messages": messages,
                "system": [
                    {
                        "text": system_prompt_text
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=BEDROCK_LLM_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract text from response - handle different possible formats
            if 'output' in response_body and 'message' in response_body['output']:
                response_text = response_body['output']['message']['content'][0]['text']
            elif 'content' in response_body:
                response_text = response_body['content'][0]['text']
            else:
                raise ValueError(f"Unexpected response format: {response_body}")
            
            # Strip markdown code blocks if present
            if response_text.startswith('```'):
                # Remove opening ```json or similar
                response_text = response_text.split('\n', 1)[1]
                # Remove closing ```
                response_text = response_text.rsplit('\n```', 1)[0]
            
            # Parse the JSON response
            extracted_json = json.loads(response_text)
            extracted_data = pydantic_model(**extracted_json)
            items = getattr(extracted_data, config["extraction_field"])
            
            print(f"    [+] Extracted {len(items)} items from {filename}")
            
            return {
                "filename": filename,
                "field_name": config["extraction_field"],
                "items": items,
                "config": config
            }
        except Exception as e:
            print(f"    [-] Error extracting from {filename}: {str(e)}")
            return {}
    
    def process_structured_data(self):
        """Process structured PDF documents and return extracted data."""
        print("\n[*] Processing structured data...")
        extracted_results = []
        
        for filename in os.listdir(self.structured_dir):
            if filename.endswith(PDF_EXTENSION):
                result = self._extract_structured_from_file(filename)
                if result:
                    extracted_results.append(result)
        
        print(f"[+] Total structured files processed: {len(extracted_results)}")
        return extracted_results
    
    def save_structured_data(self, extracted_results):
        """✅ READS scraper JSON + EXTENDS with PDF data."""
        # Load EXISTING scraper data (alumni, fellows, schools)
        existing_data = {}
        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"📂 Loaded existing scraper data: {len(existing_data)} sections")
            except Exception as e:
                print(f"⚠️  Could not load existing JSON: {e}")
        
        # Add PDF extracted data
        pdf_data = {}
        for result in extracted_results:
            field_name = result["field_name"]
            items = result["items"]
            pdf_data[field_name] = [item.model_dump() for item in items]
        
        # ✅ MERGE: existing (scraper) + new (PDF)
        combined_data = {**existing_data, **pdf_data}
        
        # Write combined data
        with open(self.output_json, "w", encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"[+] Structured data saved to {self.output_json}")
        print(f"[+] Combined data saved to {self.output_json}")
        print(f"    📊 Scraper sections: {len(existing_data)}")
        print(f"    📊 PDF sections: {len(pdf_data)}")
        print(f"    📊 TOTAL sections: {len(combined_data)}")
    
    def convert_to_documents(self, extracted_results):
        """Convert extracted structured data to Document objects."""
        print("\n[*] Converting structured data to documents...")
        documents = []
        
        for result in extracted_results:
            filename = result["filename"]
            field_name = result["field_name"]
            items = result["items"]
            config = result["config"]
            metadata_fields = config["metadata_fields"]
            
            for item in items:
                item_dict = item.model_dump()
                
                # Create document content
                content = "\n".join([f"{key}: {value}" for key, value in item_dict.items()])
                
                # Create metadata
                metadata = {"source": filename, "type": field_name}
                for field in metadata_fields:
                    if field in item_dict:
                        metadata[field] = str(item_dict[field])
                
                documents.append(Document(page_content=content, metadata=metadata))
        
        print(f"[+] Converted {len(documents)} items to Document objects")
        return documents
    
    def create_faiss_index(self, documents):
        """Create and save FAISS index from documents."""
        print(f"\n[*] Creating FAISS index at {self.faiss_dir}...")
        
        db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        db.save_local(self.faiss_dir)
        
        print(f"[+] FAISS index created with {len(documents)} documents")
        return db
    
    def load_faiss_index(self):
        """Load existing FAISS index."""
        print(f"\n[*] Loading existing FAISS index from {self.faiss_dir}...")
        
        db = FAISS.load_local(
            folder_path=self.faiss_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"[+] FAISS index loaded with {db.index.ntotal} documents")
        return db
    
    def rebuild_index(self):
        """Force rebuild the FAISS index (delete and recreate)."""
        print("\n[!] Rebuilding FAISS index...")
        
        # Delete existing index if present
        if self._is_index_exists():
            shutil.rmtree(self.faiss_dir, ignore_errors=True)
            print(f"[+] Deleted old FAISS index at {self.faiss_dir}")
        
        # Process documents
        print("\n[!] Reprocessing all documents...")
        unstructured_chunks = self.process_unstructured_data()
        extracted_results = self.process_structured_data()
        
        # Save and convert structured data
        self.save_structured_data(extracted_results)
        structured_documents = self.convert_to_documents(extracted_results)
        
        # Combine and create new index
        all_documents = unstructured_chunks + structured_documents
        print(f"\n[+] Total documents: {len(all_documents)}")
        db = self.create_faiss_index(all_documents)
        
        # Print summary
        self._print_summary(db)
        return db
    
    def run(self):
        """Main preprocessing pipeline."""
        print("="*80)
        print("Document Preprocessing Pipeline")
        print("="*80)
        
        # Check if index exists
        if self._is_index_exists():
            print("\n[!] FAISS index already exists. Skipping preprocessing.")
            db = self.load_faiss_index()
        else:
            print("\n[!] FAISS index not found. Creating new index...")
            
            # Process documents
            unstructured_chunks = self.process_unstructured_data()
            extracted_results = self.process_structured_data()
            
            # Save and convert structured data
            self.save_structured_data(extracted_results)
            structured_documents = self.convert_to_documents(extracted_results)
            
            # Combine and create index
            all_documents = unstructured_chunks + structured_documents
            print(f"\n[+] Total documents: {len(all_documents)}")
            db = self.create_faiss_index(all_documents)
        
        # Print summary
        self._print_summary(db)
        return db
    
    def _print_summary(self, db):
        """Print preprocessing summary."""
        print("\n" + "="*80)
        print("Preprocessing Summary")
        print("="*80)
        print(f"FAISS Index Location: {self.faiss_dir}")
        print(f"Total Documents in Index: {db.index.ntotal}")
        print(f"Structured Data File: {self.output_json}")
        print("="*80)
        print("[+] Ready for querying!")
        print("="*80)


# ==================== Main ====================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Document Preprocessing Pipeline - Build FAISS vector store from PDFs"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild FAISS index (delete and recreate even if it exists)"
    )
    parser.add_argument(
        "--faiss-dir",
        type=str,
        default=None,
        help="Directory to store FAISS index (default: <project-root>/public/vector-store)"
    )
    parser.add_argument(
        "--unstructured-dir",
        type=str,
        default=None,
        help="Directory containing unstructured PDFs (default: <project-root>/data/unstructured)"
    )
    parser.add_argument(
        "--structured-dir",
        type=str,
        default=None,
        help="Directory containing structured PDFs (default: <project-root>/data/structured)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file for structured data (default: <project-root>/all_structured_data.json)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    preprocessor = DocumentPreprocessor(
        unstructured_dir=args.unstructured_dir,
        structured_dir=args.structured_dir,
        faiss_dir=args.faiss_dir,
        output_json=args.output_json
    )
    
    if args.rebuild:
        print("\n[!] Rebuild flag detected. Forcing rebuild...")
        db = preprocessor.rebuild_index()
    else:
        db = preprocessor.run()

