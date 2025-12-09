#!/usr/bin/env python3
"""
TFN Structured PDF Preprocessor - Extracts staff/contacts/partners → JSON
Saves: public/json/structured_data.json (PDF data ONLY)

Usage: python scripts/structured_preprocessor.py
"""

import os
import json
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser  # ✅ REQUIRED
import boto3
from langchain_community.document_loaders import PyPDFLoader



# ==================== Pydantic Models (SAME) ====================
class StaffMember(BaseModel):
    name: str = Field(description="name of the staff member")
    role: str = Field(description="role or designation of the staff member")
    bio: str = Field(description="short biography or description of the staff member")

class StaffData(BaseModel):
    staff_members: list[StaffMember] = Field(description="List of staff members")

class Contact(BaseModel):
    name: str = Field(description="name of the contact person")
    email: str = Field(description="email address")
    phone: str = Field(description="phone number")
    organization: str = Field(description="organization or company name")

class ContactData(BaseModel):
    contacts: list[Contact] = Field(description="List of contacts")

class Partner(BaseModel):
    name: str = Field(description="name of the partner organization")
    type: str = Field(description="type of partnership")
    description: str = Field(description="description of the partnership")
    contact_person: str = Field(description="main contact person")

class PartnerData(BaseModel):
    partners: list[Partner] = Field(description="List of partners")


# ==================== Configuration (SAME) ====================
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


class StructuredPreprocessor:
    """SRP: Structured PDFs → JSON ONLY"""
    
    def __init__(self, structured_dir=None, output_json=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.structured_dir = structured_dir or os.path.join(project_root, "data", "structured")
        self.output_json = output_json or os.path.join(project_root, "public", "json", "structured_data.json")
        
        # Initialize AWS Bedrock (SAME as OLD)
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
        load_dotenv(env_path)
        
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        )
        
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
    
    def _extract_structured_from_file(self, filename):  # 👈 EXACT OLD LOGIC!
        """Extract structured data from a single PDF file (WORKING OLD CODE)."""
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
            
            # Setup extraction chain (SAME OLD LOGIC)
            pydantic_model = config["pydantic_model"]
            parser = PydanticOutputParser(pydantic_object=pydantic_model)
            format_instructions = parser.get_format_instructions()
            
            system_prompt_text = config["system_prompt"].format(format_instructions=format_instructions)
            user_message = f"Text: {raw_text}"
            
            # 👈 CRITICAL: OLD WORKING Bedrock API CALL
            messages = [{"role": "user", "content": [{"text": user_message}]}]
            body = {
                "messages": messages,
                "system": [{"text": system_prompt_text}]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            # Parse response (SAME OLD LOGIC)
            response_body = json.loads(response['body'].read())
            if 'output' in response_body and 'message' in response_body['output']:
                response_text = response_body['output']['message']['content'][0]['text']
            elif 'content' in response_body:
                response_text = response_body['content'][0]['text']
            else:
                raise ValueError(f"Unexpected response format: {response_body}")
            
            # Strip markdown code blocks
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
                response_text = response_text.rsplit('\n```', 1)[0]
            
            # Parse JSON
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
        """Process all structured PDFs."""
        print("\n[*] Processing structured PDFs...")
        extracted_results = []
        
        for filename in os.listdir(self.structured_dir):
            if filename.endswith('.pdf'):
                result = self._extract_structured_from_file(filename)
                if result:
                    extracted_results.append(result)
        
        print(f"[+] Processed {len(extracted_results)} structured files")
        return extracted_results
    
    def save_structured_data(self, extracted_results):
        """Save PDF structured data ONLY (no scraper merge)."""
        pdf_data = {}
        for result in extracted_results:
            field_name = result["field_name"]
            items = result["items"]
            pdf_data[field_name] = [item.model_dump() for item in items]
        
        with open(self.output_json, "w", encoding='utf-8') as f:
            json.dump(pdf_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ PDF structured data saved: {self.output_json}")
        for section, items in pdf_data.items():
            print(f"    📄 {section}: {len(items)} items")
    
    def run(self):
        """Main extraction pipeline."""
        print("="*80)
        print("STRUCTURED PDF PREPROCESSOR")
        print("="*80)
        
        extracted_results = self.process_structured_data()
        self.save_structured_data(extracted_results)
        
        print("\n" + "="*80)
        print("✅ STRUCTURED EXTRACTION COMPLETE!")
        print(f"📄 Output: {self.output_json}")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured data from PDFs")
    parser.add_argument("--structured-dir", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    
    preprocessor = StructuredPreprocessor(
        structured_dir=args.structured_dir,
        output_json=args.output_json
    )
    preprocessor.run()
