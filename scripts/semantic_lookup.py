#!/usr/bin/env python3
"""
Semantic Search + Direct Lookup
Search structured JSON data using semantic similarity without RAG/LLM
Fast, accurate, and deterministic results
"""

import os
import json
from difflib import SequenceMatcher
from langchain_aws import BedrockEmbeddings
import boto3
from dotenv import load_dotenv
import numpy as np

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ==================== Constants ====================

BEDROCK_EMBEDDINGS_MODEL_ID = "amazon.titan-embed-text-v2:0"


class SemanticLookup:
    """Perform semantic search on structured data without LLM/RAG."""
    
    def __init__(self, json_file=None):
        """Initialize with structured data and embeddings."""
        # Get project root directory (parent of scripts folder)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default path relative to project root
        self.json_file = json_file or os.path.join(project_root, "all_structured_data.json")
        self.data = self._load_json()
        self.embeddings = self._init_embeddings()
        
        # Pre-compute embeddings for all searchable text
        self.embeddings_cache = {}
        self._build_embeddings_cache()
    
    def _load_json(self):
        """Load structured data from JSON."""
        if not os.path.exists(self.json_file):
            print(f"[-] JSON file not found: {self.json_file}")
            return {}
        
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def _init_embeddings(self):
        """Initialize AWS Bedrock embeddings."""
        try:
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
            
            embeddings = BedrockEmbeddings(
                client=bedrock_client,
                model_id=BEDROCK_EMBEDDINGS_MODEL_ID
            )
            print("[+] Embeddings initialized")
            return embeddings
        except Exception as e:
            print(f"[-] Error initializing embeddings: {str(e)}")
            return None
    
    def _build_embeddings_cache(self):
        """Pre-compute embeddings for all items in structured data."""
        print("[*] Building embeddings cache...")
        
        for category, items in self.data.items():
            if not isinstance(items, list):
                continue
            
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    # Create searchable text from item
                    searchable_texts = []
                    if "name" in item:
                        searchable_texts.append(item["name"])
                    if "email" in item:
                        searchable_texts.append(item["email"])
                    if "role" in item:
                        searchable_texts.append(item["role"])
                    if "bio" in item:
                        searchable_texts.append(item["bio"])
                    
                    # Create key for caching
                    key = f"{category}_{idx}"
                    searchable_text = " ".join(searchable_texts)
                    
                    try:
                        # Get embedding from API
                        embedding = self.embeddings.embed_query(searchable_text)
                        self.embeddings_cache[key] = {
                            "text": searchable_text,
                            "embedding": np.array(embedding),
                            "item": item,
                            "category": category
                        }
                    except Exception as e:
                        print(f"  [-] Error embedding {key}: {str(e)}")
        
        print(f"[+] Cached {len(self.embeddings_cache)} items")
    
    def _similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _string_similarity(self, str1, str2):
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def semantic_search(self, query, top_k=5, threshold=0.3):
        """
        Search using semantic similarity.
        
        Args:
            query: User's search query
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of results with scores
        """
        if not self.embeddings:
            print("[-] Embeddings not available")
            return []
        
        try:
            # Get embedding for query
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            # Calculate similarity with all cached items
            results = []
            for key, cached in self.embeddings_cache.items():
                similarity = self._similarity(query_embedding, cached["embedding"])
                
                if similarity >= threshold:
                    results.append({
                        "similarity": float(similarity),
                        "item": cached["item"],
                        "category": cached["category"]
                    })
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        
        except Exception as e:
            print(f"[-] Error in semantic search: {str(e)}")
            return []
    
    def exact_search(self, query, category=None):
        """
        Fast exact match search using string similarity.
        Better for names and specific terms.
        
        Args:
            query: Search term (e.g., "Swastika" or "swastika@example.com")
            category: Optional filter by category
        
        Returns:
            List of matching items
        """
        results = []
        categories = [category] if category else self.data.keys()
        
        for cat in categories:
            if cat not in self.data:
                continue
            
            items = self.data[cat]
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # Search in all fields
                for field, value in item.items():
                    if isinstance(value, str):
                        similarity = self._string_similarity(query, value)
                        
                        # High threshold for exact matches
                        if similarity > 0.5:
                            results.append({
                                "match_field": field,
                                "similarity": similarity,
                                "item": item,
                                "category": cat
                            })
        
        # Remove duplicates and sort
        seen = set()
        unique_results = []
        for r in results:
            item_str = json.dumps(r["item"], sort_keys=True)
            if item_str not in seen:
                seen.add(item_str)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x["similarity"], reverse=True)
        return unique_results
    
    def lookup_by_email(self, email):
        """Fast lookup by email."""
        for category, items in self.data.items():
            if not isinstance(items, list):
                continue
            
            for item in items:
                if isinstance(item, dict) and item.get("email", "").lower() == email.lower():
                    return {
                        "item": item,
                        "category": category
                    }
        
        return None
    
    def lookup_by_name(self, name, category=None):
        """Fast lookup by name."""
        categories = [category] if category else self.data.keys()
        results = []
        
        for cat in categories:
            if cat not in self.data:
                continue
            
            items = self.data[cat]
            for item in items:
                if isinstance(item, dict) and item.get("name", "").lower() == name.lower():
                    results.append({
                        "item": item,
                        "category": cat
                    })
        
        return results
    
    def filter_by_field(self, category, field, value):
        """Filter items by field value."""
        if category not in self.data:
            return []
        
        results = []
        items = self.data[category]
        
        for item in items:
            if isinstance(item, dict):
                item_value = item.get(field, "")
                if isinstance(item_value, str) and value.lower() in item_value.lower():
                    results.append(item)
        
        return results


# ==================== Example Usage ====================

def example_semantic_search():
    """Example: Semantic search for "who handles recruitment"."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Semantic Search")
    print("Query: 'who handles recruitment'")
    print("="*60)
    
    lookup = SemanticLookup()
    results = lookup.semantic_search("who handles recruitment", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {result['similarity']:.2%}")
        print(f"    Name: {result['item'].get('name')}")
        print(f"    Role: {result['item'].get('role')}")
        print(f"    Category: {result['category']}")


def example_exact_search():
    """Example: Exact search for "Swastika"."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Exact String Search")
    print("Query: 'Swastika'")
    print("="*60)
    
    lookup = SemanticLookup()
    results = lookup.exact_search("Swastika", category="staff_members")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Match Field: {result['match_field']}")
        print(f"    Name: {result['item'].get('name')}")
        print(f"    Role: {result['item'].get('role')}")


def example_email_lookup():
    """Example: Direct email lookup."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Email Lookup (Exact Match)")
    print("Query: Looking up by email address")
    print("="*60)
    
    lookup = SemanticLookup()
    
    # Try to find contacts with emails in the data
    if lookup.data.get("contacts"):
        sample_email = lookup.data["contacts"][0].get("email")
        if sample_email:
            print(f"Email: {sample_email}")
            result = lookup.lookup_by_email(sample_email)
            if result:
                print(f"Name: {result['item'].get('name')}")
                print(f"Organization: {result['item'].get('organization')}")
                print(f"Phone: {result['item'].get('phone')}")


def example_name_lookup():
    """Example: Direct name lookup."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Name Lookup (Exact Match)")
    print("Query: 'Krishna Kumar K.C.'")
    print("="*60)
    
    lookup = SemanticLookup()
    results = lookup.lookup_by_name("Krishna Kumar K.C.", category="staff_members")
    
    if results:
        for result in results:
            print(f"Name: {result['item'].get('name')}")
            print(f"Role: {result['item'].get('role')}")
            print(f"Bio: {result['item'].get('bio')}")
    else:
        print("[-] No results found")


def example_filter_by_role():
    """Example: Filter staff by role keyword."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Filter by Role Keyword")
    print("Query: Staff with 'Coordinator' in role")
    print("="*60)
    
    lookup = SemanticLookup()
    results = lookup.filter_by_field("staff_members", "role", "Coordinator")
    
    print(f"Found {len(results)} results:\n")
    for i, item in enumerate(results[:5], 1):  # Show first 5
        print(f"[{i}] {item.get('name')} - {item.get('role')}")


def example_semantic_vs_exact():
    """Compare semantic vs exact search."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Semantic vs Exact Search Comparison")
    print("Query: 'finance person'")
    print("="*60)
    
    lookup = SemanticLookup()
    
    print("\n[SEMANTIC SEARCH]")
    semantic_results = lookup.semantic_search("finance person", top_k=3)
    for i, result in enumerate(semantic_results, 1):
        print(f"[{i}] {result['item'].get('name')} - Similarity: {result['similarity']:.2%}")
    
    print("\n[EXACT SEARCH]")
    exact_results = lookup.exact_search("finance", category="staff_members")
    for i, result in enumerate(exact_results[:3], 1):
        print(f"[{i}] {result['item'].get('name')} - Match in: {result['match_field']}")


if __name__ == "__main__":
    print("\n[*] Semantic Lookup Examples - No RAG, No LLM needed!")
    print("[*] Using embeddings + direct data lookups for speed & accuracy\n")
    
    try:
        # Run examples
        example_exact_search()
        example_name_lookup()
        example_filter_by_role()
        example_semantic_search()
        example_semantic_vs_exact()
        example_email_lookup()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
    
    except Exception as e:
        print(f"\n[-] Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
