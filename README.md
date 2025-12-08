# TFN-AI RAG Project

A Retrieval-Augmented Generation (RAG) system with intelligent query routing that combines semantic search and LLM-based analysis for organizational knowledge bases.

## Features

- **Dual Query Modes**
  - **Semantic Lookup**: Fast, accurate direct data retrieval (no LLM)
  - **RAG Chain**: Context-aware LLM responses for complex questions
  
- **Multi-Document Support**
  - Unstructured PDFs (policies, handbooks, documentation)
  - Structured PDFs (staff, contacts, partners)
  
- **Vector Store**
  - FAISS index for fast similarity search
  - Pre-computed embeddings via AWS Bedrock
  
- **Intelligent Routing**
  - Automatically selects best approach for each query
  - Lookup queries → Semantic search (fast)
  - Analysis queries → RAG (intelligent)

## Project Structure

```
RAG-PROJECT-1/
├── scripts/
│   ├── preprocess_docs.py      # Build FAISS index from PDFs
│   ├── query_docs.py           # Query using RAG chain
│   ├── semantic_lookup.py      # Direct semantic search
│   └── colab*.py               # Development notebooks
├── data/
│   ├── structured/             # Staff, contacts, partners PDFs
│   └── unstructured/           # Policy, handbook PDFs
├── public/
│   ├── vector-store/           # FAISS index (index.faiss)
│   └── all_structured_data.json
├── .env                        # AWS credentials (git-ignored)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure AWS Credentials
Create a `.env` file in the project root:
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token
```

### 4. Add Documents
- Place unstructured PDFs in `data/unstructured`
- Place structured PDFs in `data/structured`

## Usage

### Preprocess Documents (First Time)
```powershell
cd scripts
python preprocess_docs.py
```

### Rebuild Index
```powershell
python preprocess_docs.py --rebuild
```

### Query with RAG
```powershell
python query_docs.py "How do I report harassment?"
```

### Interactive RAG Mode
```powershell
python query_docs.py --interactive
```

### Semantic Lookup (No LLM)
```powershell
python semantic_lookup.py
```

## How It Works

### 1. Preprocessing Pipeline
- **Unstructured Data**: PDFs → Text chunks → Embeddings → FAISS Index
- **Structured Data**: PDFs → LLM extraction → JSON → Documents → FAISS Index
- **Output**: `index.faiss` in `public/vector-store`

### 2. Query Processing

**Semantic Lookup** (Fast, Direct)
- Search `all_structured_data.json` using embeddings
- No LLM calls
- Best for: Names, emails, roles, exact matches

**RAG Chain** (Smart, Context-Aware)
- Retrieve context from FAISS index
- Send to AWS Bedrock LLM
- Generate answer with sources
- Best for: Complex questions, analysis, explanations

## Models Used

- **LLM**: `amazon.nova-lite-v1:0` (AWS Bedrock)
- **Embeddings**: `amazon.titan-embed-text-v2:0` (AWS Bedrock)

## Performance

- **First Run**: 1-2 minutes (builds index)
- **Subsequent Runs**: 1-5 seconds (loads index)
- **Semantic Search**: <1 second per query
- **RAG Query**: 2-5 seconds (includes LLM inference)

## API Integration

For Node.js app integration, use intelligent query routing:
```
Query Classification
    ↓
├─ Lookup Pattern → semanticLookup()
├─ Analysis Pattern → ragQuery()
└─ Unknown → hybridQuery()
```

## Configuration

Edit these constants in `scripts/preprocess_docs.py`:

```python
BEDROCK_LLM_MODEL_ID = "amazon.nova-lite-v1:0"
BEDROCK_EMBEDDINGS_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_EXTENSION = ".pdf"
```

## Supported Document Types

- **staff.pdf**: Extract staff members with roles and bios
- **contacts.pdf**: Extract contact information
- **partners-and-supporters.pdf**: Extract partner organizations
- **Other PDFs**: Stored as unstructured content

## Troubleshooting

### AWS Credentials Expired
- Generate new temporary credentials from AWS account
- Update `.env` file with new credentials
- Rerun preprocessing or queries

### FAISS Index Issues
- Delete `public/vector-store` folder
- Run `python preprocess_docs.py --rebuild`

### Missing Documents
- Ensure PDFs are in correct folders
- Check file naming matches configuration in `preprocess_docs.py`

## Dependencies

See `requirements.txt` for complete list. Key packages:
- `langchain` - Orchestration
- `langchain-aws` - AWS integration
- `faiss-cpu` - Vector search
- `boto3` - AWS SDK
- `pydantic` - Data validation

## License

Internal use only
