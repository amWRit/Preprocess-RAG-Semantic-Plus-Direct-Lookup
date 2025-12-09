#!/usr/bin/env python3
"""
TFN RAG Index Builder - Combines JSONs → FAISS Vector Store
Loads: scraped_data.json + structured_data.json + unstructured_chunks.json
Saves: combined_data.json + public/vector-store/index.faiss

Usage: python scripts/vector_store_builder.py --rebuild
"""

import os
import json
import argparse
import shutil
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
import boto3
from dotenv import load_dotenv


class RAGIndexBuilder:
    """SRP: JSONs → Combined JSON + FAISS Index"""
    
    def __init__(self, json_dir=None, faiss_dir=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.json_dir = json_dir or os.path.join(project_root, "public", "json")
        self.faiss_dir = faiss_dir or os.path.join(project_root, "public", "vector-store")
        self.faiss_index_path = os.path.join(self.faiss_dir, "index.faiss")
        
        # JSON paths
        self.scraped_json = os.path.join(self.json_dir, "scraped_data.json")
        self.structured_json = os.path.join(self.json_dir, "structured_data.json")
        self.unstructured_json = os.path.join(self.json_dir, "unstructured_chunks.json")
        self.combined_json = os.path.join(self.json_dir, "combined_data.json")
        
        # Initialize embeddings
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
        load_dotenv(env_path)
        
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        )
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v2:0"
        )
        
        os.makedirs(self.faiss_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
    
    def _index_exists(self):
        """Check if FAISS index exists."""
        return os.path.exists(self.faiss_index_path)
    
    def load_json(self, path):
        """Load JSON file safely."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"[!] JSON not found: {path}")
        return {}
    
    def merge_all_data(self):
        """Merge scraped + structured + unstructured."""
        print("\n[*] Loading JSON sources...")
        scraped = self.load_json(self.scraped_json)
        structured = self.load_json(self.structured_json)
        unstructured = self.load_json(self.unstructured_json)
        
        print(f"    📄 scraped_data.json: {len(scraped)} sections")
        print(f"    📄 structured_data.json: {len(structured)} sections") 
        print(f"    📄 unstructured_chunks.json: {len(unstructured)} chunks")
        
        # Merge structured data (dict of lists)
        combined = {}
        for data_dict in [scraped, structured]:
            for key, value in data_dict.items():
                if isinstance(value, list):
                    combined.setdefault(key, []).extend(value)
                else:
                    combined[key] = value
        
        # Store unstructured separately (for document conversion)
        if unstructured:
            combined['unstructured_chunks'] = unstructured
        
        # Save combined JSON
        with open(self.combined_json, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"✅ Combined data: {self.combined_json}")
        
        return combined
    
    def json_to_documents(self, combined_data):
        """Convert combined JSON → Document objects."""
        print("\n[*] Converting JSON to Documents...")
        documents = []
        
        # Structured data (scraped + structured)
        for section_name, items in combined_data.items():
            if section_name == 'unstructured_chunks':
                continue  # Handle separately
            
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        # Create content from all string fields
                        content_parts = [f"{k}: {v}" for k, v in item.items() 
                                       if isinstance(v, str) and v.strip()]
                        content = "\n".join(content_parts) if content_parts else str(item)
                        
                        metadata = {
                            "source": section_name,
                            "item_id": i,
                            "total_items": len(items)
                        }
                        # Add key fields to metadata
                        for key in ['name', 'email', 'school', 'role']:
                            if key in item:
                                metadata[key] = str(item[key])[:100]
                        
                        documents.append(Document(page_content=content, metadata=metadata))
        
        # Unstructured chunks
        if 'unstructured_chunks' in combined_data:
            for chunk in combined_data['unstructured_chunks']:
                documents.append(Document(
                    page_content=chunk['chunk'],  # 👈 FIXED: 'chunk' not 'content'
                    metadata={
                        'source': chunk['source'],
                        'page': chunk.get('page', 0),
                        'type': 'unstructured'
                    }
                ))
        
        print(f"✅ Created {len(documents)} Documents")
        return documents
    
    def create_faiss_index(self, documents):
        """Create and save FAISS index."""
        print(f"\n[*] Building FAISS index ({len(documents)} documents)...")
        db = FAISS.from_documents(documents, self.embeddings)
        db.save_local(self.faiss_dir)
        print(f"✅ FAISS index saved: {self.faiss_dir}")
        return db
    
    def load_faiss_index(self):
        """Load existing FAISS index."""
        print(f"[*] Loading existing FAISS index: {self.faiss_dir}")
        db = FAISS.load_local(
            self.faiss_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded index: {db.index.ntotal} vectors")
        return db
    
    def run(self, rebuild=False):
        """Main pipeline: JSONs → FAISS."""
        print("="*80)
        print("RAG INDEX BUILDER")
        print("="*80)
        
        if rebuild or not self._index_exists():
            print("\n[*] Building NEW index...")
            combined = self.merge_all_data()
            documents = self.json_to_documents(combined)
            db = self.create_faiss_index(documents)
        else:
            print("\n[*] Loading EXISTING index...")
            db = self.load_faiss_index()
        
        print("\n" + "="*80)
        print("INDEX SUMMARY")
        print("="*80)
        print(f"📁 FAISS: {self.faiss_dir}")
        print(f"📊 Vectors: {db.index.ntotal}")
        print(f"📄 Combined JSON: {self.combined_json}")
        print("="*80)
        print("✅ RAG READY!")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG index from JSON sources")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--json-dir", type=str, default=None, help="JSON directory")
    parser.add_argument("--faiss-dir", type=str, default=None, help="FAISS directory")
    args = parser.parse_args()
    
    builder = RAGIndexBuilder(
        json_dir=args.json_dir,
        faiss_dir=args.faiss_dir
    )
    builder.run(rebuild=args.rebuild)
