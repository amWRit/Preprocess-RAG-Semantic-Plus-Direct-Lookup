#!/usr/bin/env python3
"""
🚀 TFN RAG PIPELINE ORCHESTRATOR
Smartly runs scraper → preprocessors → vector store with caching

Usage:
    python scripts/pipeline.py --rebuild
    python scripts/pipeline.py --preprocess-only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def clean_outputs(mode=None, rebuild=False):
    """Delete only relevant output JSONs based on mode."""
    files = {
        'scraped': "public/json/scraped_data.json",
        'structured': "public/json/structured_data.json",
        'unstructured': "public/json/unstructured_chunks.json",
        'combined': "public/json/combined_data.json"
    }
    to_delete = []
    
    if rebuild:
        to_delete = list(files.values())
    elif mode == 'scrape_only':
        to_delete = [files['scraped']]
    elif mode == 'preprocess_only':
        to_delete = [files['structured'], files['unstructured']]
    
    for json_path in to_delete:
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
                print(f"🧹 DELETED: {json_path}")
            except Exception as e:
                print(f"⚠️  Could not delete {json_path}: {e}")
        else:
            print(f"ℹ️  {json_path} not found (already clean)")


def run_scraper():
    """🌐 Run web scraper tool."""
    print("🌐 [*] Running scraper...")
    result = subprocess.run([
        sys.executable, "-m", "scripts.scraper",  # 👈 UPDATED!
        "--scrape-all"
    ], capture_output=False, text=True)
    
    if result.returncode != 0:
        print("❌ Scraper failed!")
        return False
    print("✅ Scraper complete! → scraped_data.json")
    return True


def run_preprocessor():
    """📄 Run structured + unstructured preprocessors."""
    print("📄 [*] Running preprocessors...")
    
    # Structured PDFs → structured_data.json
    print("    📋 Structured preprocessor...")
    cmd_structured = [sys.executable, "-m", "scripts.structured_preprocessor"]
    result_structured = subprocess.run(cmd_structured, capture_output=False, text=True)
    if result_structured.returncode != 0:
        print("❌ Structured preprocessor failed!")
        return False
    print("    ✅ Structured complete!")
    
    # Unstructured PDFs → unstructured_chunks.json
    print("    📄 Unstructured preprocessor...")
    cmd_unstructured = [sys.executable, "-m", "scripts.unstructured_preprocessor"]
    result_unstructured = subprocess.run(cmd_unstructured, capture_output=False, text=True)
    if result_unstructured.returncode != 0:
        print("❌ Unstructured preprocessor failed!")
        return False
    print("    ✅ Unstructured complete!")
    
    return True


def run_indexbuilder(rebuild=False):
    """🔗 Run index builder tool."""
    print("🔗 [*] Running index builder...")
    cmd = [sys.executable, "-m", "scripts.indexbuilder"]  # updated here
    if rebuild:
        cmd.append("--rebuild")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print("❌ Index builder failed!")
        return False
    print("✅ Index ready! → public/vector-store/")
    return True

def check_prerequisites():
    """📁 Ensure all directories exist."""
    required_dirs = [
        "data/unstructured",
        "data/structured",
        "public/json",
        "public/vector-store"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created: {dir_path}")
    
    if missing:
        print(f"📁 Created {len(missing)} directories")
    
    # PDF status
    structured_pdfs = list(Path("data/structured").glob("*.pdf"))
    unstructured_pdfs = list(Path("data/unstructured").glob("*.pdf"))
    
    print(f"📁 {len(structured_pdfs)} structured PDFs")
    print(f"📁 {len(unstructured_pdfs)} unstructured PDFs")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="🚀 TFN RAG Pipeline")
    parser.add_argument("--rebuild", action="store_true", 
                       help="Force rebuild vector store (ignores cache)")
    parser.add_argument("--scrape-only", action="store_true", 
                       help="🌐 Scraper only")
    parser.add_argument("--preprocess-only", action="store_true", 
                       help="📄 Preprocessors only")
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 TFN RAG PIPELINE")
    print("="*80)
    
    # cleaning
    if args.scrape_only:
        mode = 'scrape_only'
    elif args.preprocess_only:
        mode = 'preprocess_only'
    else:
        mode = 'full'  # Default full pipeline

    clean_outputs(mode=mode, rebuild=args.rebuild)
    
    if not check_prerequisites():
        return 1
    
    success = True
    
    if args.scrape_only:
        success = run_scraper()
    elif args.preprocess_only:
        success = run_preprocessor()
    else:
        # Full RAG pipeline
        success = (run_scraper() and 
                  run_preprocessor() and 
                  run_indexbuilder(args.rebuild))
    
    if success:
        print("\n🎉 PIPELINE COMPLETE!")
        print("📊 Data ready in public/json/ & public/vector-store/")
        print("="*80)
        return 0
    else:
        print("\n❌ PIPELINE FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
