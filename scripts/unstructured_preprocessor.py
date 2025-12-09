#!/usr/bin/env python3
"""
TFN Unstructured PDF Preprocessor - Chunks PDFs → JSON
Saves: public/json/unstructured_chunks.json

Usage: python scripts/unstructured_preprocessor.py
"""

import os
import json
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class UnstructuredPreprocessor:
    """SRP: Unstructured PDFs → JSON Chunks ONLY"""
    
    def __init__(self, unstructured_dir=None, output_json=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.unstructured_dir = unstructured_dir or os.path.join(project_root, "data", "unstructured")
        self.output_json = output_json or os.path.join(project_root, "public", "json", "unstructured_chunks.json")
        
        # Text splitter
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
    
    def process_unstructured_data(self):
        """Process all unstructured PDFs → chunks."""
        print("\n[*] Processing unstructured PDFs...")
        all_chunks = []
        
        pdf_count = 0
        for filename in os.listdir(self.unstructured_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.unstructured_dir, filename)
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    chunks = self.splitter.split_documents(documents)
                    
                    for chunk in chunks:
                        all_chunks.append({
                            "source": filename,
                            "chunk": chunk.page_content,
                            "page": chunk.metadata.get("page", 0)
                        })
                    
                    pdf_count += 1
                    print(f"    [+] {filename}: {len(documents)} pages → {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"    [-] Error processing {filename}: {str(e)}")
        
        print(f"[+] Total: {len(all_chunks)} chunks from {pdf_count} PDFs")
        return all_chunks
    
    def save_unstructured_data(self, all_chunks):
        """Save chunks to JSON."""
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Unstructured chunks saved: {self.output_json}")
        print(f"    📊 {len(all_chunks)} total chunks")
    
    def run(self):
        """Main extraction pipeline."""
        print("="*80)
        print("UNSTRUCTURED PDF PREPROCESSOR")
        print("="*80)
        
        all_chunks = self.process_unstructured_data()
        self.save_unstructured_data(all_chunks)
        
        print("\n" + "="*80)
        print("✅ UNSTRUCTURED PROCESSING COMPLETE!")
        print(f"📄 Output: {self.output_json}")
        print("="*80)


# ==================== CLI Entrypoint ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process unstructured PDFs to JSON chunks")
    parser.add_argument("--unstructured-dir", type=str, default=None,
                       help="Directory containing unstructured PDFs")
    parser.add_argument("--output-json", type=str, default=None,
                       help="Output JSON file for chunks")
    args = parser.parse_args()
    
    preprocessor = UnstructuredPreprocessor(
        unstructured_dir=args.unstructured_dir,
        output_json=args.output_json
    )
    preprocessor.run()
