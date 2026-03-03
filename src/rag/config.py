import os
from pathlib import Path

# Google Sheets ID for benchmark data and run results (sheets.py)
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "")

# Chunking strategies to evaluate (pipeline.py)
STRATEGIES = [
    "markdown-optimized",
]

# Embedding models to evaluate (pipeline.py, embedding.py)
EMBEDDING_MODELS = [
    "BAAI/bge-base-en-v1.5",
]

# Reranker models to evaluate; None means no reranking (pipeline.py, reranker.py)
RERANKER_MODELS = [
    "BAAI/bge-reranker-base",
]

# Search type: "vector" (pure near_vector) or "hybrid" (BM25 + vector fusion)
SEARCH_TYPE = "vector"

# Hybrid search alpha: 0 = pure BM25, 1 = pure vector, 0.75 = vector-weighted
HYBRID_ALPHA = 0.5

# Query transformation: "none" or "hyde" (Hypothetical Document Embeddings)
QUERY_TRANSFORM = "none"

# Values of k to retrieve from vector search, swept in evaluation (pipeline.py)
RETRIEVAL_K_VALUES = [250]

# Number of chunks to keep after reranking and pass to the LLM (pipeline.py)
TOP_K = 20

# Benchmark queries to run against each strategy (pipeline.py)
QUERIES = [
    "What is the latest Weaviate version?",
    "How do I install Weaviate?",
    "What is the Weaviate license?",
    "What is weaviate embedded mode?",
    "How does HNSW indexing work in Weaviate?",
    "What is multi-tenancy in Weaviate?",
    "How do I create and restore backups in Weaviate?",
    "How does hybrid search work in Weaviate?",
    "How does RBAC work in Weaviate?",
    "How does replication work in Weaviate?",
    "How do I use generative search (RAG) in Weaviate?",
    "How do I filter search results in Weaviate?",
]

# Directory for cached documents, chunks, and embeddings (load.py, pipeline.py)
CACHE_DIR = Path("cache")

# Max recursion depth for inlining nested MDX includes (parsing.py)
MAX_INCLUDE_DEPTH = 3

# System prompt sent to the LLM for RAG responses (llm.py, pipeline.py)
SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Weaviate documentation.
You will be given a question and a list of chunks taken from the Weaviate documentation.
You will need to answer the question using ONLY the information provided in the chunks.
"""
