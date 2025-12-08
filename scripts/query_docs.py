#!/usr/bin/env python3
"""
TFN-AI RAG Query Script
Query the pre-built FAISS vector store for document retrieval and RAG interactions

Usage:
    python query_docs.py "Your question here"
    python query_docs.py --interactive

Requirements:
    pip install langchain-aws langchain langchain-community faiss-cpu python-dotenv
"""

import os
import sys
import json
import argparse
import boto3
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_classic import hub
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Load environment variables from .env.local file (from parent directory)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.local'))

console = Console()


# ==================== Initialize AWS & RAG Components ====================

def initialize_rag():
    """Initialize AWS Bedrock, embeddings, and RAG chain."""
    
    # Initialize AWS Bedrock client
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

    # Initialize embeddings
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

    return bedrock_client, embeddings


def load_faiss_index(faiss_dir="../public/vector-store", embeddings=None):
    """Load FAISS index from disk."""
    
    if not os.path.isabs(faiss_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_dir = os.path.abspath(os.path.join(script_dir, faiss_dir))
    
    if not os.path.exists(faiss_dir):
        console.print(f"[red][-] FAISS index not found at {faiss_dir}[/red]")
        console.print("[yellow][!] Please run 'python preprocess_docs.py' first to create the index.[/yellow]")
        return None
    
    try:
        db = FAISS.load_local(
            folder_path=faiss_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        console.print(f"[green][+] Loaded FAISS index from {faiss_dir} with {db.index.ntotal} documents.[/green]")
        return db
    except Exception as e:
        console.print(f"[red][-] Error loading FAISS index: {str(e)}[/red]")
        return None


def setup_rag_chain(db, bedrock_client):
    """Setup RAG chain with retriever and Bedrock LLM."""
    
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Create a custom LLM function that calls Bedrock directly
    def invoke_bedrock(input_text):
        """Call Bedrock directly with proper message formatting."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": input_text
                    }
                ]
            }
        ]
        
        body = {
            "messages": messages,
            "system": [
                {
                    "text": "You are a helpful assistant. Use the provided context to answer questions accurately and concisely. If you don't know the answer based on the context, say so."
                }
            ]
        }
        
        try:
            response = bedrock_client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text from response
            if 'output' in response_body and 'message' in response_body['output']:
                response_text = response_body['output']['message']['content'][0]['text']
            else:
                response_text = "Unable to generate response"
            
            return response_text
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Build RAG chain using the custom Bedrock function
    def format_docs(docs):
        """Format retrieved documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: f"Context:\n{x['context']}\n\nQuestion: {x['question']}")
        | RunnableLambda(invoke_bedrock)
    )
    
    return rag_chain


def query_rag(rag_chain, question):
    """Execute a query on the RAG chain and return response."""
    
    try:
        with console.status("[bold green]Generating response..."):
            response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


def format_response(question, response, query_num=1):
    """Format and display query response with rich formatting."""
    
    console.print(f"\n[bold yellow]Q{query_num}:[/bold yellow] {question}")
    console.print(Panel(
        Text(response, style="#36312F"),
        title=f"Response {query_num}",
        border_style="green",
        padding=(1, 2),
        expand=False
    ))
    console.print("[dim]" + "─" * 80 + "[/dim]")


def interactive_mode(rag_chain):
    """Interactive query mode - allows user to enter queries in a loop."""
    
    console.print(Panel(
        Text("Interactive RAG Query Mode", style="bold cyan", justify="center"),
        border_style="bright_blue",
        padding=(1, 2)
    ))
    console.print("[yellow]Enter your questions (type 'exit' or 'quit' to exit):[/yellow]\n")
    
    query_num = 1
    while True:
        try:
            question = console.input("[bold cyan]>> [/bold cyan]").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Exiting interactive mode.[/yellow]")
                break
            
            if not question:
                continue
            
            response = query_rag(rag_chain, question)
            format_response(question, response, query_num)
            query_num += 1
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Exiting.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


def batch_mode(rag_chain, queries):
    """Batch query mode - process multiple predefined queries."""
    
    console.print(Panel(
        Text("RAG Batch Query Mode", style="bold cyan", justify="center"),
        border_style="bright_blue",
        padding=(1, 2)
    ))
    
    for i, question in enumerate(queries, 1):
        response = query_rag(rag_chain, question)
        format_response(question, response, i)


def show_index_stats(db):
    """Display statistics about the FAISS index."""
    
    stats_table = Table(title="FAISS Index Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    stats_table.add_row("Total Documents", str(db.index.ntotal))
    stats_table.add_row("Index Dimension", str(db.index.d) if hasattr(db.index, 'd') else "N/A")
    
    console.print(stats_table)


def parse_arguments():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Query pre-built FAISS vector store for RAG interactions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_docs.py "Who is Swastika Shrestha?"
  python query_docs.py --interactive
  python query_docs.py --stats
  python query_docs.py --query "How do I report harassment?" --faiss-dir my_index
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string to search (if not provided with --interactive, enters interactive mode)"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter interactive query mode"
    )
    
    parser.add_argument(
        "--faiss-dir",
        type=str,
        default="../public/vector-store",
        help="Directory containing FAISS index (default: ../public/vector-store)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show FAISS index statistics"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)"
    )
    
    return parser.parse_args()


# ==================== Main ====================

def main():
    """Main entry point for query script."""
    
    args = parse_arguments()
    
    # Initialize RAG components
    console.print("[*] Initializing AWS Bedrock and embeddings...")
    try:
        bedrock_client, embeddings = initialize_rag()
        console.print("[+] AWS Bedrock initialized successfully.")
    except Exception as e:
        console.print(f"[red][-] Failed to initialize AWS Bedrock: {str(e)}[/red]")
        console.print("[yellow][!] Check your .env file for valid AWS credentials.[/yellow]")
        sys.exit(1)

    # Load FAISS index
    console.print(f"[*] Loading FAISS index from {args.faiss_dir}...")
    db = load_faiss_index(args.faiss_dir, embeddings)
    if db is None:
        sys.exit(1)

    # Show stats if requested
    if args.stats:
        show_index_stats(db)
        return

    # Setup RAG chain
    console.print("[*] Setting up RAG chain...")
    rag_chain = setup_rag_chain(db, bedrock_client)
    console.print("[+] RAG chain initialized successfully.\n")

    # Determine mode based on arguments
    if args.interactive or (args.query is None):
        # Interactive mode
        interactive_mode(rag_chain)
    else:
        # Single query mode
        response = query_rag(rag_chain, args.query)
        format_response(args.query, response, 1)


if __name__ == "__main__":
    main()
