#!/usr/bin/env python3
"""
🚀 SIMPLE PIPELINE RUNNER
AUTO-DELETES public/all_structured_data.json → flexible_scraper → preprocessdocs
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

def clean_structured_data():
    """🧹 Delete public/all_structured_data.json before pipeline."""
    json_path = "public/all_structured_data.json"
    
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
            print(f"🧹 DELETED: {json_path}")
        except Exception as e:
            print(f"⚠️  Could not delete {json_path}: {e}")
    else:
        print(f"ℹ️  {json_path} not found (already clean)")

def run_scraper():
    """Run flexible scraper."""
    print("🌐 [*] Running flexible scraper...")
    result = subprocess.run([
        sys.executable, "-m", "scripts.flexible_scraper", 
        "--scrape-all"
    ], capture_output=False, text=True)
    
    if result.returncode != 0:
        print("❌ Scraper failed!")
        return False
    print("✅ Scraper complete!")
    return True

def run_preprocessor(rebuild=False):
    """Run your proven preprocessor."""
    print("📄 [*] Running preprocessor...")
    cmd = [sys.executable, "-m", "scripts.preprocess_docs"]
    
    if rebuild:
        cmd.append("--rebuild")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print("❌ Preprocessor failed!")
        return False
    print("✅ Preprocessor complete!")
    return True

def check_prerequisites():
    """Check required directories/files exist."""
    required_dirs = [
        "data/unstructured",
        "data/structured",
        "public/vector-store"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created: {dir_path}")
    
    if missing:
        print(f"📁 Created missing directories: {missing}")
    
    # Check for PDF files
    structured_pdfs = list(Path("data/structured").glob("*.pdf"))
    unstructured_pdfs = list(Path("data/unstructured").glob("*.pdf"))
    
    print(f"📁 Found {len(structured_pdfs)} structured PDFs")
    print(f"📁 Found {len(unstructured_pdfs)} unstructured PDFs")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="🚀 Simple Pipeline Runner")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--scrape-only", action="store_true", help="Only run scraper")
    parser.add_argument("--preprocess-only", action="store_true", help="Only run preprocessor")
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 SIMPLE PIPELINE RUNNER")
    print("="*80)
    
    # 🧹 AUTO-CLEAN structured data
    clean_structured_data()
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    success = True
    
    # Pipeline phases
    if args.scrape_only:
        success = run_scraper()
    elif args.preprocess_only:
        success = run_preprocessor(args.rebuild)
    else:
        # Full pipeline
        success = run_scraper() and run_preprocessor(args.rebuild)
    
    if success:
        print("\n🎉 PIPELINE COMPLETE!")
        print("="*80)
        return 0
    else:
        print("\n❌ PIPELINE FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
