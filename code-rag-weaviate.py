#!/usr/bin/env python3
"""
Code RAG - Retrieval Augmented Generation for code search using Weaviate and Voyage AI embeddings.

This script provides functionality for chunking source code, indexing those chunks in a
Weaviate vector database, and retrieving relevant code snippets based on semantic similarity.

Usage:
    python code-rag-weaviate.py --codebase <codebase> chunker <path-to-source-tree> <output-json-file>
    python code-rag-weaviate.py --codebase <codebase> indexer <chunk-list-json-file>
    python code-rag-weaviate.py --codebase <codebase> retrieve <user prompt for vector similarity search> [-k <num-results>] [-g] [-f] [-s] [-e|--no-entity-filter] [-o <output-file>]
    python code-rag-weaviate.py --codebase <codebase> retrieve -q <query-file.txt> [-k <num-results>] [-g] [-f] [-s] [-e|--no-entity-filter] [-o <output-file>]
    python code-rag-weaviate.py --codebase <codebase> resetdb

Example:
    # For xv6 codebase
    python code-rag-weaviate.py --codebase xv6 chunker ../xv6-riscv xv6-chunks.json
    python code-rag-weaviate.py --codebase xv6 indexer xv6-chunks.json
    python code-rag-weaviate.py --codebase xv6 retrieve "how does file system initialization work?"
    python code-rag-weaviate.py --codebase xv6 retrieve -f "how does file system initialization work?"
    python code-rag-weaviate.py --codebase xv6 retrieve -s "how does file system initialization work?"
    python code-rag-weaviate.py --codebase xv6 retrieve -s -e "how does file system initialization work?"  # Use both summary and entity filtering
    python code-rag-weaviate.py --codebase xv6 retrieve -s --no-entity-filter "how does file system initialization work?"  # Use only summary filtering
    python code-rag-weaviate.py --codebase xv6 retrieve "how does file system initialization work?" -o results.json  # Save results to JSON file
    python code-rag-weaviate.py --codebase xv6 retrieve -q query.txt -o results.json  # Load query from file and save results to JSON
    python code-rag-weaviate.py --codebase xv6 resetdb
    
    # For llama_index codebase
    python code-rag-weaviate.py --codebase llama_index chunker ./data/llama_index llama-chunks.json
    python code-rag-weaviate.py --codebase llama_index indexer llama-chunks.json
    python code-rag-weaviate.py --codebase llama_index retrieve "how does the embedding API work?"
    python code-rag-weaviate.py --codebase llama_index retrieve -g "explain the retriever implementation"
    python code-rag-weaviate.py --codebase llama_index retrieve -g -f "explain the retriever implementation"
    python code-rag-weaviate.py --codebase llama_index retrieve -g -s "explain the retriever implementation"
    python code-rag-weaviate.py --codebase llama_index retrieve -g -s -e "explain the retriever implementation"
    python code-rag-weaviate.py --codebase llama_index retrieve -g "explain the retriever implementation" -o retriever_answer.txt  # Save only the answer to text file
    python code-rag-weaviate.py --codebase llama_index retrieve "explain the retriever implementation" -o retriever_chunks.json  # Save chunks to JSON file

Options:
    --codebase: Specify which codebase to use (xv6 or llama_index)
    -q, --query-file: Path to a text file containing the query
    -k, --top-k: Number of results to return (default: 5)
    -g, --generate: Generate an answer using OpenRouter DeepSeek Coder LLM
    -f, --filter: Filter retrieved chunks to remove irrelevant code
    -s, --summary: Use file summaries to improve retrieval
    -e, --entity-filter: Use entity-based filtering for functions and classes (default: enabled)
    --no-entity-filter: Disable entity-based filtering
    -o, --output: Path to save output file. When used with -g, saves only the answer as text.
                  Without -g, saves retrieved chunks as JSON.
"""

import os
import sys
import json
import uuid
import argparse
import requests
from pathlib import Path
import importlib.util
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
from weaviate.util import generate_uuid5
from dotenv import load_dotenv
from rich import print

# Load environment variables from .env file
load_dotenv()

# Import the CodeSplitter
try:
    # Using importlib to handle the code-meta.py import
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_meta_path = os.path.join(script_dir, "code-meta.py")
    
    spec = importlib.util.spec_from_file_location("code_meta", code_meta_path)
    code_meta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_meta)
    CodeSplitter = code_meta.CodeSplitter
    
    from llama_index.core.schema import Document
    from llama_index.embeddings.voyageai import VoyageEmbedding
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure you have the required packages installed:")
    print("pip install llama-index tree-sitter tree-sitter-languages weaviate-client python-dotenv")
    print("pip install llama-index-embeddings-voyageai")
    sys.exit(1)

# Constants

# Codebase-specific configurations
CODEBASE_CONFIGS = {
    "xv6": {
        "class_name": "Xv6CodeChunk",
        "summary_class_name": "Xv6Summary",
        "chunk_lines": 60,  # Reduced from 60 for more focused chunks
        "chunk_lines_overlap": 5,  # Increased from 5 for better context preservation
        "max_chars": 2048  # Reduced from 2048 for more focused chunks
    },
    "llama_index": {
        "class_name": "LlamaIndexCodeChunk",
        "summary_class_name": "LlamaIndexSummary",
        "chunk_lines": 500,  # Reduced from 500 for more granular retrieval
        "chunk_lines_overlap": 1,  # Increased from 1 for better context preservation
        "max_chars": 16384  # Reduced from 16384 for more focused chunks
    }
}

# Default to xv6 if not specified
DEFAULT_CODEBASE = "xv6"

class VoyageVectorizer:
    """Voyage AI embeddings handler for Weaviate."""
    
    def __init__(self):
        # Check for API key in environment
        if "VOYAGE_API_KEY" not in os.environ:
            print("Warning: VOYAGE_API_KEY environment variable not set.")
            print("Please set your Voyage API key in the .env file: VOYAGE_API_KEY='your-api-key'")
            sys.exit(1)
        
        # Create a VoyageEmbedding instance
        self.voyage_embedding = VoyageEmbedding(
            voyage_api_key=os.environ["VOYAGE_API_KEY"],
            model_name="voyage-3"
        )
    
    def get_embedding(self, text):
        """Get embedding for a single text string."""
        return self.voyage_embedding.get_text_embedding(text)

import os
import sys
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

# … your existing CODEBASE_CONFIGS, DEFAULT_CODEBASE …

def get_weaviate_client():
    if "WEAVIATE_URL" not in os.environ or "WEAVIATE_API_KEY" not in os.environ:
        print("Please set WEAVIATE_URL and WEAVIATE_API_KEY")
        sys.exit(1)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )
    if not client.is_ready():
        print("Weaviate client not ready")
        sys.exit(1)
    print("Weaviate cloud client is ready")
    return client

def setup_weaviate_schema(client, codebase=DEFAULT_CODEBASE):
    """
    Create (if missing) the two collections for code chunks and summaries—
    using TEXT_ARRAY for all list‑of‑string fields.
    """
    cfg         = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
    chunk_cls   = cfg["class_name"]
    summary_cls = cfg["summary_class_name"]

    common_props = [
        Property(name="filepath", data_type=DataType.TEXT),
        Property(name="filename", data_type=DataType.TEXT),
        Property(name="relpath",  data_type=DataType.TEXT),
    ]

    # 1) Code‑chunk collection
    if chunk_cls not in client.collections.list_all():
        client.collections.create(
            chunk_cls,
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=common_props + [
                Property(name="content",                 data_type=DataType.TEXT),
                Property(name="start_line",              data_type=DataType.INT),
                Property(name="end_line",                data_type=DataType.INT),
                Property(name="length",                  data_type=DataType.INT),
                Property(name="language",                data_type=DataType.TEXT),
                Property(name="chunking_method",         data_type=DataType.TEXT),
                Property(name="chunk_function_names",    data_type=DataType.TEXT_ARRAY),  # was list
                Property(name="chunk_class_names",       data_type=DataType.TEXT_ARRAY),
                Property(name="document_function_names", data_type=DataType.TEXT_ARRAY),
                Property(name="document_class_names",    data_type=DataType.TEXT_ARRAY),
            ]
        )
        print(f"Created collection: {chunk_cls}")

    # 2) Summary collection
    if summary_cls not in client.collections.list_all():
        client.collections.create(
            summary_cls,
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=common_props + [
                Property(name="summary", data_type=DataType.TEXT),
            ]
        )
        print(f"Created collection: {summary_cls}")

    return chunk_cls, summary_cls

def reset_db(codebase=DEFAULT_CODEBASE):
    """
    Delete & recreate the two collections (v4 API).
    """
    client = get_weaviate_client()
    cfg    = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
    chunk_cls, summary_cls = cfg["class_name"], cfg["summary_class_name"]

    for cls in (chunk_cls, summary_cls):
        try:
            client.collections.delete(cls)
            print(f"Deleted collection: {cls}")
        except Exception:
            pass

    setup_weaviate_schema(client, codebase)
    client.close()
    print("Reset complete.")


import os
import sys
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

# … your existing CODEBASE_CONFIGS, DEFAULT_CODEBASE …

def get_weaviate_client():
    if "WEAVIATE_URL" not in os.environ or "WEAVIATE_API_KEY" not in os.environ:
        print("Please set WEAVIATE_URL and WEAVIATE_API_KEY")
        sys.exit(1)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )
    if not client.is_ready():
        print("Weaviate client not ready")
        sys.exit(1)
    print("Weaviate cloud client is ready")
    return client

def setup_weaviate_schema(client, codebase=DEFAULT_CODEBASE):
    """
    Create (if missing) the two collections for code chunks and summaries—
    using TEXT_ARRAY for all list‑of‑string fields.
    """
    cfg         = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
    chunk_cls   = cfg["class_name"]
    summary_cls = cfg["summary_class_name"]

    common_props = [
        Property(name="filepath", data_type=DataType.TEXT),
        Property(name="filename", data_type=DataType.TEXT),
        Property(name="relpath",  data_type=DataType.TEXT),
    ]

    # 1) Code‑chunk collection
    if chunk_cls not in client.collections.list_all():
        client.collections.create(
            chunk_cls,
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=common_props + [
                Property(name="content",                 data_type=DataType.TEXT),
                Property(name="start_line",              data_type=DataType.INT),
                Property(name="end_line",                data_type=DataType.INT),
                Property(name="length",                  data_type=DataType.INT),
                Property(name="language",                data_type=DataType.TEXT),
                Property(name="chunking_method",         data_type=DataType.TEXT),
                Property(name="chunk_function_names",    data_type=DataType.TEXT_ARRAY),  # was list
                Property(name="chunk_class_names",       data_type=DataType.TEXT_ARRAY),
                Property(name="document_function_names", data_type=DataType.TEXT_ARRAY),
                Property(name="document_class_names",    data_type=DataType.TEXT_ARRAY),
            ]
        )
        print(f"Created collection: {chunk_cls}")

    # 2) Summary collection
    if summary_cls not in client.collections.list_all():
        client.collections.create(
            summary_cls,
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
            properties=common_props + [
                Property(name="summary", data_type=DataType.TEXT),
            ]
        )
        print(f"Created collection: {summary_cls}")

    return chunk_cls, summary_cls

def reset_db(codebase=DEFAULT_CODEBASE):
    """
    Delete & recreate the two collections (v4 API).
    """
    client = get_weaviate_client()
    cfg    = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
    chunk_cls, summary_cls = cfg["class_name"], cfg["summary_class_name"]

    for cls in (chunk_cls, summary_cls):
        try:
            client.collections.delete(cls)
            print(f"Deleted collection: {cls}")
        except Exception:
            pass

    setup_weaviate_schema(client, codebase)
    client.close()
    print("Reset complete.")

# import os
# import sys
# import weaviate
# from weaviate.classes.init import Auth
# from weaviate.classes.config import Configure, Property, DataType

# Use your existing CODEBASE_CONFIGS and DEFAULT_CODEBASE here

# def get_weaviate_client():
#     """Instantiate and verify a Weaviate v4 cloud client."""
#     if "WEAVIATE_URL" not in os.environ or "WEAVIATE_API_KEY" not in os.environ:
#         print("Please set both WEAVIATE_URL and WEAVIATE_API_KEY in your environment")
#         sys.exit(1)

#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url=os.environ["WEAVIATE_URL"],
#         auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
#     )
#     if not client.is_ready():
#         print("Weaviate cloud client not ready — check your URL/API key")
#         sys.exit(1)
#     print("Weaviate cloud client is ready")
#     return client

# def setup_weaviate_schema(client, codebase=DEFAULT_CODEBASE):
#     """
#     v4: Create (if missing) two collections:
#       - <CodeChunkClass>
#       - <SummaryClass>
#     using Configure.Vectorizer.text2vec_weaviate() and the Property/DataType API.
#     """
#     cfg          = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
#     chunk_cls    = cfg["class_name"]
#     summary_cls  = cfg["summary_class_name"]

#     # shared props
#     common_props = [
#         Property(name="filepath", data_type=DataType.TEXT),
#         Property(name="filename", data_type=DataType.TEXT),
#         Property(name="relpath",  data_type=DataType.TEXT),
#     ]

#     # 1) code‑chunk collection
#     if chunk_cls not in client.collections.list_all():
#         client.collections.create(
#             chunk_cls,
#             vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
#             properties=common_props + [
#                 Property(name="content",                data_type=DataType.TEXT),
#                 Property(name="start_line",             data_type=DataType.INT),
#                 Property(name="end_line",               data_type=DataType.INT),
#                 Property(name="length",                 data_type=DataType.INT),
#                 Property(name="language",               data_type=DataType.TEXT),
#                 Property(name="chunking_method",        data_type=DataType.TEXT),
#                 Property(name="chunk_function_names",   data_type=[DataType.TEXT]),
#                 Property(name="chunk_class_names",      data_type=[DataType.TEXT]),
#                 Property(name="document_function_names",data_type=[DataType.TEXT]),
#                 Property(name="document_class_names",   data_type=[DataType.TEXT]),
#             ]
#         )
#         print(f"Created collection: {chunk_cls}")

#     # 2) summary collection
#     if summary_cls not in client.collections.list_all():
#         client.collections.create(
#             summary_cls,
#             vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
#             properties=common_props + [
#                 Property(name="summary", data_type=DataType.TEXT),
#             ]
#         )
#         print(f"Created collection: {summary_cls}")

#     return chunk_cls, summary_cls

# def reset_db(codebase=DEFAULT_CODEBASE):
#     """
#     v4: Delete and then recreate the two collections for the given codebase.
#     """
#     client = get_weaviate_client()
#     cfg    = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
#     chunk_cls, summary_cls = cfg["class_name"], cfg["summary_class_name"]

#     # delete if exists
#     for cls in (chunk_cls, summary_cls):
#         try:
#             client.collections.delete(cls)
#             print(f"Deleted collection: {cls}")
#         except Exception:
#             # ignore if it wasn't there
#             pass

#     # recreate
#     setup_weaviate_schema(client, codebase)
#     client.close()
#     print("Reset complete.")

# def get_weaviate_client():
#     """Get or create a Weaviate cloud client."""
#     # Check for API keys in environment
#     if "WEAVIATE_URL" not in os.environ:
#         print("Warning: WEAVIATE_URL environment variable not set.")
#         print("Please set your Weaviate URL in the .env file: WEAVIATE_URL='your-weaviate-url'")
#         sys.exit(1)
        
#     if "WEAVIATE_API_KEY" not in os.environ:
#         print("Warning: WEAVIATE_API_KEY environment variable not set.")
#         print("Please set your Weaviate API key in the .env file: WEAVIATE_API_KEY='your-api-key'")
#         sys.exit(1)
    
#     # Get API keys from environment
#     weaviate_url = os.environ["WEAVIATE_URL"]
#     weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
    
#     # Create a cloud-based client
#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url=weaviate_url,
#         auth_credentials=Auth.api_key(weaviate_api_key),
#     )
    
#     # Check if the client was created successfully
#     if client.is_ready():
#         print("Weaviate cloud client is ready")
#     else:
#         print("Error creating Weaviate cloud client")
#         sys.exit(1)
    
#     return client

# def setup_weaviate_schema(client, codebase=DEFAULT_CODEBASE):
#     """v4: create the two collections for code chunks and summaries."""
#     cfg = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
#     chunk_cls   = cfg["class_name"]
#     summary_cls = cfg["summary_class_name"]

#     # Define the shared property specs
#     common_props = [
#         Property(name="filepath",   data_type=DataType.TEXT),
#         Property(name="filename",   data_type=DataType.TEXT),
#         Property(name="relpath",    data_type=DataType.TEXT),
#     ]

#     # 1) create the code‑chunk collection
#     if chunk_cls not in client.collections.list_all():
#         client.collections.create(
#             chunk_cls,
#             vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
#             properties=common_props + [
#                 Property(name="content",          data_type=DataType.TEXT),
#                 Property(name="start_line",       data_type=DataType.INT),
#                 Property(name="end_line",         data_type=DataType.INT),
#                 Property(name="length",           data_type=DataType.INT),
#                 Property(name="language",         data_type=DataType.TEXT),
#                 Property(name="chunking_method",  data_type=DataType.TEXT),
#                 Property(name="chunk_function_names",  data_type=[DataType.TEXT]),
#                 Property(name="chunk_class_names",     data_type=[DataType.TEXT]),
#                 Property(name="document_function_names", data_type=[DataType.TEXT]),
#                 Property(name="document_class_names",    data_type=[DataType.TEXT]),
#             ]
#         )
#         print(f"Created collection: {chunk_cls}")
#     if summary_cls not in client.collections.list_all():
#         client.collections.create(
#             summary_cls,
#             vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
#             properties=common_props + [
#                 Property(name="summary", data_type=DataType.TEXT),
#             ]
#         )
#         print(f"Created collection: {summary_cls}")

#     return chunk_cls, summary_cls

#def setup_weaviate_schema(client, voyage_vectorizer, codebase=DEFAULT_CODEBASE):
#    """Setup Weaviate schema for code chunks and summaries."""
#    # Get the class names for the specified codebase
#    class_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["class_name"]
#    summary_class_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["summary_class_name"]
#    
#    # Check if classes already exist
#    try:
#        existing_classes = client.schema.get()["classes"] or []
#        existing_class_names = [cls["class"] for cls in existing_classes]
#        
#        # Create code chunks class if it doesn't exist
#        if class_name not in existing_class_names:
#            print(f"Creating new class: {class_name}")
#            
#            # Define properties for code chunks
#            properties = [
#                {"name": "content", "dataType": ["text"]},
#                {"name": "filepath", "dataType": ["text"]},
#                {"name": "filename", "dataType": ["text"]},
#                {"name": "relpath", "dataType": ["text"]},
#                {"name": "start_line", "dataType": ["int"]},
#                {"name": "end_line", "dataType": ["int"]},
#                {"name": "length", "dataType": ["int"]},
#                {"name": "language", "dataType": ["text"]},
#                {"name": "chunking_method", "dataType": ["text"]},
#                {"name": "chunk_function_names", "dataType": ["text[]"]},
#                {"name": "chunk_class_names", "dataType": ["text[]"]},
#                {"name": "document_function_names", "dataType": ["text[]"]},
#                {"name": "document_class_names", "dataType": ["text[]"]}
#            ]
#            
#            # Create the class
#            client.schema.create_class({
#                "class": class_name,
#                "description": f"Code chunks for {codebase} codebase",
#                "vectorizer": "none",  # We'll use our custom Voyage AI embeddings
#                "properties": properties
#            })
#            print(f"Created {class_name} class")
#        
#        # Create summaries class if it doesn't exist
#        if summary_class_name not in existing_class_names:
#            print(f"Creating new class: {summary_class_name}")
#            
#            # Define properties for summaries
#            summary_properties = [
#                {"name": "summary", "dataType": ["text"]},
#                {"name": "filepath", "dataType": ["text"]},
#                {"name": "filename", "dataType": ["text"]},
#                {"name": "relpath", "dataType": ["text"]}
#            ]
#            
#            # Create the class
#            client.schema.create_class({
#                "class": summary_class_name,
#                "description": f"File summaries for {codebase} codebase",
#                "vectorizer": "none",  # We'll use our custom Voyage AI embeddings
#                "properties": summary_properties
#            })
#            print(f"Created {summary_class_name} class")
#        
#    except Exception as e:
#        print(f"Error setting up Weaviate schema: {e}")
#        sys.exit(1)
#    
#    return class_name, summary_class_name

def find_source_files(directory):
    """Find all .c and .h files in directory and subdirectories."""
    source_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.c', '.h', '.py')):
                source_files.append(os.path.join(root, file))
    return source_files

def extract_function_names(code, language):
    """Extract function names from code based on language."""
    function_names = []
    
    if language == "c":
        # Simple regex for C function definitions
        # This is a basic implementation and might miss some edge cases
        import re
        # Look for function definitions like: return_type function_name(params)
        pattern = r'\b\w+\s+(\w+)\s*\([^;]*\)\s*{'
        matches = re.findall(pattern, code)
        function_names = [name for name in matches if name not in ['if', 'for', 'while', 'switch']]
    
    elif language == "python":
        # Simple regex for Python function definitions
        import re
        # Look for "def function_name("
        pattern = r'def\s+(\w+)\s*\('
        function_names = re.findall(pattern, code)
    
    return function_names

def extract_class_names(code, language):
    """Extract class names from code based on language."""
    class_names = []
    
    if language == "c":
        # C doesn't have classes in the traditional sense
        # Look for struct definitions as a proxy
        import re
        pattern = r'struct\s+(\w+)\s*{'
        class_names = re.findall(pattern, code)
    
    elif language == "python":
        # Simple regex for Python class definitions
        import re
        pattern = r'class\s+(\w+)'
        class_names = re.findall(pattern, code)
    
    return class_names

def extract_comments(code, language):
    """Extract comments from code based on language."""
    comments = []
    
    if language == "c":
        # Extract C-style comments
        import re
        # Single line comments: // comment
        single_line = re.findall(r'\/\/(.+)', code)
        # Multi-line comments: /* comment */
        multi_line = re.findall(r'\/\*(.+?)\*\/', code, re.DOTALL)
        comments = single_line + multi_line
    
    elif language == "python":
        # Extract Python comments
        import re
        # Single line comments: # comment
        single_line = re.findall(r'#(.+)', code)
        # Docstrings: """ comment """ or ''' comment '''
        docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
        docstrings += re.findall(r"'''(.+?)'''", code, re.DOTALL)
        comments = single_line + docstrings
    
    return [comment.strip() for comment in comments if comment.strip()]

def create_fallback_splitter(language):
    """Create a simple line-based splitter as fallback when AST parsing fails."""
    from llama_index.core.node_parser import SentenceSplitter
    
    # Configure based on language
    if language == "c":
        # C code often has longer lines but fewer of them
        chunk_size = 1000
        chunk_overlap = 200
    else:  # python or other
        # Python tends to be more verbose with more lines
        chunk_size = 1500
        chunk_overlap = 300
    
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_delimiter="\n"
    )
def process_file(file_path, base_dir, codebase=DEFAULT_CODEBASE):
    """Process a source file and return its chunks with enhanced metadata and summary."""
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        
        # Skip empty files
        if not code.strip():
            print(f"Skipping empty file: {file_path}")
            return [], None
        
        # Configure the language based on file extension
        language = "c"
        if file_path.endswith(".py"):
            language = "python"
        
        # Extract code entities for enhanced metadata
        function_names = extract_function_names(code, language)
        class_names = extract_class_names(code, language)
        comments = extract_comments(code, language)
        
        # Generate summary using OpenRouter
        print(f"Generating summary for {file_path}")
        summary = generate_summary(code)
        
        # Extract filename and relative path
        filename = os.path.basename(file_path)
        relpath = os.path.relpath(file_path, base_dir)
        
        # Create summary object
        summary_object = {
            "filename": filename,
            "path": file_path,
            "relpath": relpath,
            "summary": summary
        }
        
        # Create a Document from the code with enhanced metadata
        document = Document(
            text=code,
            metadata={
                "filepath": file_path,
                "base_dir": base_dir,
                "codebase": codebase,
                "language": language,
                "function_names": function_names,
                "class_names": class_names,
                "has_comments": len(comments) > 0,
                "filename": filename,
                "relpath": relpath
            }
        )
        
        # Get codebase-specific configuration
        config = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
        
        # Configure the CodeSplitter
        code_splitter = CodeSplitter(
            language=language,
            chunk_lines=config["chunk_lines"],
            chunk_lines_overlap=config["chunk_lines_overlap"],
            max_chars=config["max_chars"],
        )
 
        # Try AST-based chunking first
        nodes = []
        ast_chunking_successful = True
        try:
            nodes = code_splitter.get_nodes_from_documents([document])
            if not nodes:
                print(f"AST chunking produced no nodes for {file_path}, falling back to line-based chunking")
                ast_chunking_successful = False
        except Exception as e:
            print(f"AST chunking failed for {file_path}: {e}, falling back to line-based chunking")
            ast_chunking_successful = False
        
        # If AST chunking failed, use fallback splitter
        if not ast_chunking_successful:
            fallback_splitter = create_fallback_splitter(language)
            nodes = fallback_splitter.get_nodes_from_documents([document])
            print(f"Fallback chunking produced {len(nodes)} nodes")
        
        # Convert nodes to the required format with enhanced metadata
        chunks = []
        for node in nodes:
            # Extract chunk-specific code entities
            chunk_text = node.text
            chunk_function_names = extract_function_names(chunk_text, language)
            chunk_class_names = extract_class_names(chunk_text, language)
            
            # Create enhanced chunk with both document-level and chunk-level metadata
            chunk = {
                # File and location metadata
                "filepath": node.metadata.get("filepath", ""),
                "filename": node.metadata.get("filename", ""),
                "relpath": node.metadata.get("relpath", ""),
                "start_line": node.metadata.get("start_line", 0),
                "end_line": node.metadata.get("end_line", 0),
                "length": node.metadata.get("end_line", 0) - node.metadata.get("start_line", 0) + 1,
                
                # General metadata
                "language": language,
                "codebase": codebase,
                "chunking_method": "ast" if ast_chunking_successful else "fallback",
                
                # Chunk-specific code entities (only in this chunk)
                "chunk_function_names": chunk_function_names,
                "chunk_class_names": chunk_class_names,
                
                # Document-level code entities (from the entire file)
                "document_function_names": function_names,
                "document_class_names": class_names,
                
                # Content
                "content": chunk_text
            }
            chunks.append(chunk)
        
        print(f"Processed {file_path}: Found {len(function_names)} functions, {len(class_names)} classes")
        return chunks, summary_object
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def chunk_source_tree(source_dir, output_file, codebase=DEFAULT_CODEBASE):
    """Process all files in the source directory and save chunks and summaries to JSON files."""
    # Check if the directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        sys.exit(1)
    
    print(f"Using codebase configuration: {codebase}")
    
    # Get the absolute path
    abs_dir_path = os.path.abspath(source_dir)
    
    # Find all source files
    source_files = find_source_files(abs_dir_path)
    print(f"Found {len(source_files)} source files in {abs_dir_path}")
    
    # Process each file and collect chunks and summaries
    all_chunks = []
    all_summaries = []
    for file_path in source_files:
        print(f"Processing {file_path}")
        chunks, summary_object = process_file(file_path, abs_dir_path, codebase)
        
        # Skip if no chunks were generated (empty file or error)
        if not chunks:
            print(f"  - No chunks generated for {file_path}")
            continue
            
        all_chunks.extend(chunks)
        if summary_object:
            all_summaries.append(summary_object)
        print(f"  - Generated {len(chunks)} chunks")
    
    # Save the chunks to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Save the summaries to a separate JSON file
    summaries_output_file = "code_summaries.json"
    with open(summaries_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\nSuccessfully processed {len(source_files)} files.")
    print(f"Generated {len(all_chunks)} chunks.")
    print(f"Generated {len(all_summaries)} file summaries.")
    print(f"Results saved to {output_file}")
    print(f"Summaries saved to {summaries_output_file}")

# def index_chunks(chunk_file, codebase=DEFAULT_CODEBASE):
#     """Index chunks and summaries into Weaviate with Voyage AI embeddings."""
#     # Load chunks from JSON
#     try:
#         with open(chunk_file, 'r') as f:
#             chunks = json.load(f)
#     except Exception as e:
#         print(f"Error loading chunks from {chunk_file}: {e}")
#         sys.exit(1)
    
#     print(f"Loaded {len(chunks)} chunks from {chunk_file}")
#     print(f"Using codebase configuration: {codebase}")
    
#     # Setup Weaviate with Voyage AI embeddings
#     voyage_vectorizer = VoyageVectorizer()
#     client = get_weaviate_client()
#     class_name, summary_class_name = setup_weaviate_schema(client , codebase)
    
#     # Load summaries from code_summaries.json
#     try:
#         with open("code_summaries.json", 'r') as f:
#             summaries = json.load(f)
#         print(f"Loaded {len(summaries)} summaries from code_summaries.json")
#     except Exception as e:
#         print(f"Error loading summaries from code_summaries.json: {e}")
#         print("Will proceed without summaries")
#         summaries = []
    
#     # Index summaries
#     if summaries:
#         print(f"Indexing {len(summaries)} file summaries...")
#         for i, summary in enumerate(summaries):
#             try:
#                 # Generate UUID5 based on the file path for consistency
#                 summary_id = generate_uuid5(summary["path"])
                
#                 # Create the summary object with embedding
#                 summary_embedding = voyage_vectorizer.get_embedding(summary["summary"])
                
#                 # Prepare data object
#                 summary_data = {
#                     "summary": summary["summary"],
#                     "filepath": summary["path"],
#                     "filename": summary["filename"],
#                     "relpath": summary["relpath"]
#                 }
                
#                 # Add to Weaviate
#                 client.data_object.create(
#                     data_object=summary_data,
#                     class_name=summary_class_name,
#                     uuid=summary_id,
#                     vector=summary_embedding
#                 )
                
#                 if i % 10 == 0:
#                     print(f"Indexed {i+1}/{len(summaries)} summaries...")
                    
#             except Exception as e:
#                 print(f"Error indexing summary {i}: {e}")
        
#         print(f"Successfully indexed {len(summaries)} summaries in Weaviate class '{summary_class_name}'")
    
#     # Index chunks in batches
#     print(f"Indexing {len(chunks)} chunks with enhanced metadata...")
#     batch_size = 50
    
#     with client.batch as batch:
#         batch.batch_size = batch_size
        
#         for i, chunk in enumerate(chunks):
#             # Generate UUID5 based on file path and line numbers for consistency
#             chunk_id = generate_uuid5(f"{chunk['filepath']}:{chunk['start_line']}-{chunk['end_line']}")
            
#             # Create the embedding
#             content_embedding = voyage_vectorizer.get_embedding(chunk["content"])
            
#             # Prepare data object with all the metadata
#             data_object = {
#                 "content": chunk["content"],
#                 "filepath": chunk["filepath"],
#                 "filename": chunk["filename"],
#                 "relpath": chunk.get("relpath", ""),
#                 "start_line": chunk["start_line"],
#                 "end_line": chunk["end_line"],
#                 "length": chunk["length"],
#                 "language": chunk.get("language", ""),
#                 "chunking_method": chunk.get("chunking_method", "unknown")
#             }
            
#             # Add chunk-specific function and class names as arrays
#             if "chunk_function_names" in chunk and chunk["chunk_function_names"]:
#                 if isinstance(chunk["chunk_function_names"], list):
#                     data_object["chunk_function_names"] = chunk["chunk_function_names"]
#                 else:
#                     data_object["chunk_function_names"] = [chunk["chunk_function_names"]]
            
#             if "chunk_class_names" in chunk and chunk["chunk_class_names"]:
#                 if isinstance(chunk["chunk_class_names"], list):
#                     data_object["chunk_class_names"] = chunk["chunk_class_names"]
#                 else:
#                     data_object["chunk_class_names"] = [chunk["chunk_class_names"]]
            
#             # Add document-level function and class names as arrays
#             if "document_function_names" in chunk and chunk["document_function_names"]:
#                 if isinstance(chunk["document_function_names"], list):
#                     data_object["document_function_names"] = chunk["document_function_names"]
#                 else:
#                     data_object["document_function_names"] = [chunk["document_function_names"]]
            
#             if "document_class_names" in chunk and chunk["document_class_names"]:
#                 if isinstance(chunk["document_class_names"], list):
#                     data_object["document_class_names"] = chunk["document_class_names"]
#                 else:
#                     data_object["document_class_names"] = [chunk["document_class_names"]]
            
#             # Add to batch
#             batch.add_data_object(
#                 data_object=data_object,
#                 class_name=class_name,
#                 uuid=chunk_id,
#                 vector=content_embedding
#             )
            
#             if i % 100 == 0:
#                 print(f"Added {i+1}/{len(chunks)} chunks to batch...")
    
#     # Get total count of indexed objects
#     try:
#         count = client.query.aggregate(class_name).with_meta_count().do()
#         count_value = count["data"]["Aggregate"][class_name][0]["meta"]["count"]
#         print(f"Successfully indexed {count_value} chunks in Weaviate class '{class_name}'")
#     except Exception as e:
#         print(f"Error getting count: {e}")
#         print("Chunks were indexed, but count could not be verified")
# def index_chunks(chunk_file, codebase=DEFAULT_CODEBASE):
#     """Index chunks and summaries into Weaviate with Voyage AI embeddings (v4 simple loop)."""
#     # 1) Load chunks JSON
#     try:
#         with open(chunk_file, 'r') as f:
#             chunks = json.load(f)
#     except Exception as e:
#         print(f"Error loading chunks from {chunk_file}: {e}")
#         sys.exit(1)
#     print(f"Loaded {len(chunks)} chunks from {chunk_file}")
#     print(f"Using codebase configuration: {codebase}")

#     # 2) Setup vectorizer & client & schema
#     voyage_vectorizer = VoyageVectorizer()
#     client = get_weaviate_client()
#     class_name, summary_class_name = setup_weaviate_schema(client, codebase)

#     # 3) Load summaries JSON
#     try:
#         with open("code_summaries.json", 'r') as f:
#             summaries = json.load(f)
#         print(f"Loaded {len(summaries)} summaries from code_summaries.json")
#     except Exception as e:
#         print(f"Error loading summaries: {e}")
#         summaries = []

#     # 4) Index summaries one by one
#     if summaries:
#         print(f"Indexing {len(summaries)} file summaries...")
#         summary_coll = client.collections.get(summary_class_name)
#         for summary in summaries:
#             sid = generate_uuid5(summary["path"])
#             vec = voyage_vectorizer.get_embedding(summary["summary"])
#             props = {
#                 "summary": summary["summary"],
#                 "filepath": summary["path"],
#                 "filename": summary["filename"],
#                 "relpath": summary["relpath"]
#             }
#             summary_coll.data.insert(properties=props, vector=vec, uuid=sid)
#         print(f"Indexed {len(summaries)} summaries into '{summary_class_name}'")

#     # 5) Index chunks one by one
#     print(f"Indexing {len(chunks)} chunks with enhanced metadata...")
#     chunk_coll = client.collections.get(class_name)
#     for chunk in chunks:
#         cid = generate_uuid5(f"{chunk['filepath']}:{chunk['start_line']}-{chunk['end_line']}")
#         vec = voyage_vectorizer.get_embedding(chunk["content"])
#         props = {
#             "content": chunk["content"],
#             "filepath": chunk["filepath"],
#             "filename": chunk["filename"],
#             "relpath": chunk.get("relpath", ""),
#             "start_line": chunk["start_line"],
#             "end_line": chunk["end_line"],
#             "length": chunk["length"],
#             "language": chunk.get("language", ""),
#             "chunking_method": chunk.get("chunking_method", "unknown"),
#             **({"chunk_function_names":    chunk["chunk_function_names"]}    if chunk.get("chunk_function_names")    else {}),
#             **({"chunk_class_names":       chunk["chunk_class_names"]}       if chunk.get("chunk_class_names")       else {}),
#             **({"document_function_names": chunk["document_function_names"]} if chunk.get("document_function_names") else {}),
#             **({"document_class_names":    chunk["document_class_names"]}    if chunk.get("document_class_names")    else {}),
#         }
#         chunk_coll.data.insert(properties=props, vector=vec, uuid=cid)
#     print(f"Indexed {len(chunks)} chunks into '{class_name}'")

#     # 6) Optionally verify count
#     try:
#         agg = client.query.aggregate(class_name).with_meta_count().do()
#         count = agg["data"]["Aggregate"][class_name][0]["meta"]["count"]
#         print(f"Total objects in '{class_name}': {count}")
#     except Exception:
#         pass

#     client.close()
import json
import sys
from weaviate.exceptions import UnexpectedStatusCodeError

def index_chunks(chunk_file, codebase=DEFAULT_CODEBASE):
    """Index chunks and summaries into Weaviate with Voyage AI embeddings (v4 simple loop + upsert)."""
    # 1) Load chunks JSON
    try:
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error loading chunks from {chunk_file}: {e}")
        sys.exit(1)
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")
    print(f"Using codebase configuration: {codebase}")

    # 2) Setup vectorizer & client & schema
    voyage_vectorizer = VoyageVectorizer()
    client            = get_weaviate_client()
    class_name, summary_class_name = setup_weaviate_schema(client, codebase)

    # 3) Load summaries JSON
    try:
        with open("code_summaries.json", 'r') as f:
            summaries = json.load(f)
        print(f"Loaded {len(summaries)} summaries from code_summaries.json")
    except Exception as e:
        print(f"Error loading summaries: {e}")
        summaries = []

    # 4) Index summaries with insert→update fallback
    if summaries:
        print(f"Indexing {len(summaries)} file summaries (upsert)...")
        summary_coll = client.collections.get(summary_class_name)
        for summary in summaries:
            sid  = generate_uuid5(summary["path"])
            vec  = voyage_vectorizer.get_embedding(summary["summary"])
            props = {
                "summary": summary["summary"],
                "filepath": summary["path"],
                "filename": summary["filename"],
                "relpath": summary["relpath"],
            }
            try:
                summary_coll.data.insert(properties=props, vector=vec, uuid=sid)
            except UnexpectedStatusCodeError as e:
                if "already exists" in str(e):
                    # update both props & vector if the object already exists :contentReference[oaicite:0]{index=0}
                    summary_coll.data.update(uuid=sid, properties=props, vector=vec)
                else:
                    raise
        print(f"Upserted {len(summaries)} summaries into '{summary_class_name}'")

    # 5) Index chunks one by one (same upsert logic)
    print(f"Indexing {len(chunks)} chunks with enhanced metadata (upsert)...")
    chunk_coll = client.collections.get(class_name)
    for chunk in chunks:
        cid  = generate_uuid5(f"{chunk['filepath']}:{chunk['start_line']}-{chunk['end_line']}")
        vec  = voyage_vectorizer.get_embedding(chunk["content"])
        props = {
            "content": chunk["content"],
            "filepath": chunk["filepath"],
            "filename": chunk["filename"],
            "relpath": chunk.get("relpath", ""),
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "length": chunk["length"],
            "language": chunk.get("language", ""),
            "chunking_method": chunk.get("chunking_method", "unknown"),
            **({"chunk_function_names":    chunk["chunk_function_names"]}    if chunk.get("chunk_function_names")    else {}),
            **({"chunk_class_names":       chunk["chunk_class_names"]}       if chunk.get("chunk_class_names")       else {}),
            **({"document_function_names": chunk["document_function_names"]} if chunk.get("document_function_names") else {}),
            **({"document_class_names":    chunk["document_class_names"]}    if chunk.get("document_class_names")    else {}),
        }
        try:
            chunk_coll.data.insert(properties=props, vector=vec, uuid=cid)
        except UnexpectedStatusCodeError as e:
            if "already exists" in str(e):
                chunk_coll.data.update(uuid=cid, properties=props, vector=vec)
            else:
                raise
    print(f"Upserted {len(chunks)} chunks into '{class_name}'")

    # 6) Optionally verify count
    try:
        agg   = client.query.aggregate(class_name).with_meta_count().do()
        count = agg["data"]["Aggregate"][class_name][0]["meta"]["count"]
        print(f"Total objects in '{class_name}': {count}")
    except Exception:
        pass

    client.close()

def filter_chunk_content(chunk, query):
    """Filter irrelevant data from a chunk using OpenRouter LLM."""
    # Check for API key in environment
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Warning: OPENROUTER_API_KEY environment variable not set.")
        print("Please set your OpenRouter API key in the .env file: OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Create the prompt
    prompt = f"""You are an expert code assistant that helps filter irrelevant code.
I'll provide you with a code chunk and a query. Your task is to remove any irrelevant parts of the code
that don't relate to the query, while preserving the structure and format of the code.
Only remove code that is completely irrelevant to the query.
Keep all function signatures, important comments, and structural elements.
Return ONLY the filtered code content, nothing else.

QUERY: {query}

CODE CHUNK (from {chunk['filename']}, lines {chunk['start_line']}-{chunk['end_line']}):
{chunk['content']}

FILTERED CODE:"""
    
    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "HTTP-Referer": "https://code-rag-example.com",  # Replace with your actual domain
        "X-Title": "Code RAG"  # Optional site title for rankings
    }
    
    payload = {
        "model": "openai/gpt-3.5-turbo",  # Using a known working model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    
    # Make the API request
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract and return the filtered content
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            filtered_content = result["choices"][0]["message"]["content"]
            return filtered_content
        else:
            print("Warning: No filtered content generated, returning original content.")
            return chunk['content']
    
    except Exception as e:
        print(f"Error filtering chunk content: {e}")
        return chunk['content']  # Return original content on error

def extract_entities_from_query(query):
    """Extract specific code entities from a query.
    
    Only extracts:
    1. Functions/methods (words followed by parentheses)
    2. File names (words with extensions)
    3. Special delimiters like __init__
    
    If none of these are found, returns an empty list to skip filtering.
    """
    import re
    
    # Extract function/method names (words followed by parentheses)
    function_pattern = r'\b(\w+)\s*\('
    function_names = re.findall(function_pattern, query)
    
    # Extract file names (words with common code file extensions)
    # Expanded to include more common extensions
    file_pattern = r'\b(\w+\.(c|h|py|js|ts|java|cpp|hpp|cs|go|rb|php|html|css|xml|json))\b'
    file_matches = re.findall(file_pattern, query)
    file_names = [match[0] for match in file_matches]  # Extract just the filename
    
    # Extract special delimiters like __init__, __main__, etc.
    delimiter_pattern = r'\b(__\w+__)\b'
    delimiters = re.findall(delimiter_pattern, query)
    
    # Combine all specific entities
    entities = list(set(function_names + file_names + delimiters))
    
    if entities:
        print(f"Extracted specific code entities: {', '.join(entities)}")
    else:
        print("No specific code entities found in query, skipping entity-based filtering")
    
    return entities

def filter_chunks(query, chunks, codebase=DEFAULT_CODEBASE):
    """Filter irrelevant data from chunks using LLM.
    
    Args:
        query: The query string for filtering
        chunks: The chunks to filter
        codebase: The codebase being used
        
    Returns:
        List of filtered chunks
    """
    print(f"Filtering {len(chunks)} chunks to remove irrelevant code...")
    
    # Filter each chunk using the LLM
    filtered_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Filtering chunk {i+1}/{len(chunks)}...")
        filtered_content = filter_chunk_content(chunk, query)
        
        # Create a new chunk with the filtered content
        filtered_chunk = chunk.copy()
        filtered_chunk["content"] = filtered_content
        filtered_chunks.append(filtered_chunk)
    
    print(f"Successfully filtered {len(filtered_chunks)} chunks.")
    return filtered_chunks

def generate_answer(query, chunks):
    """Generate an answer using OpenRouter LLM based on retrieved chunks with enhanced context."""
    # Check for API key in environment
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Warning: OPENROUTER_API_KEY environment variable not set.")
        print("Please set your OpenRouter API key in the .env file: OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Extract codebase information
    codebase = chunks[0].get('codebase', 'unknown') if chunks else 'unknown'
    language = chunks[0].get('language', 'unknown') if chunks else 'unknown'
    
    # Format the context from chunks with enhanced metadata
    context = ""
    for i, chunk in enumerate(chunks):
        # Add file and function context
        file_info = f"File: {chunk['filename']}"
        line_info = f"Lines {chunk['start_line']}-{chunk['end_line']}"
        
        # Add function/class information if available
        chunk_entity_info = ""
        doc_entity_info = ""
        
        # Chunk-level entities
        if 'chunk_function_names' in chunk and chunk['chunk_function_names']:
            if isinstance(chunk['chunk_function_names'], list):
                chunk_entity_info += f"Functions in chunk: {', '.join(chunk['chunk_function_names'])}"
            else:
                chunk_entity_info += f"Functions in chunk: {chunk['chunk_function_names']}"
        
        if 'chunk_class_names' in chunk and chunk['chunk_class_names']:
            if isinstance(chunk['chunk_class_names'], list):
                chunk_entity_info += f" | Classes in chunk: {', '.join(chunk['chunk_class_names'])}"
            else:
                chunk_entity_info += f" | Classes in chunk: {chunk['chunk_class_names']}"
        
        # Document-level entities
        if 'document_function_names' in chunk and chunk['document_function_names']:
            if isinstance(chunk['document_function_names'], list):
                doc_entity_info += f"All functions in file: {', '.join(chunk['document_function_names'])}"
            else:
                doc_entity_info += f"All functions in file: {chunk['document_function_names']}"
        
        if 'document_class_names' in chunk and chunk['document_class_names']:
            if isinstance(chunk['document_class_names'], list):
                doc_entity_info += f" | All classes in file: {', '.join(chunk['document_class_names'])}"
            else:
                doc_entity_info += f" | All classes in file: {chunk['document_class_names']}"
                
        # For backward compatibility
        if not chunk_entity_info and 'function_names' in chunk and chunk['function_names']:
            if isinstance(chunk['function_names'], list):
                chunk_entity_info += f"Functions: {', '.join(chunk['function_names'])}"
            else:
                chunk_entity_info += f"Functions: {chunk['function_names']}"
        
        if not chunk_entity_info and 'class_names' in chunk and chunk['class_names']:
            if isinstance(chunk['class_names'], list):
                chunk_entity_info += f" | Classes: {', '.join(chunk['class_names'])}"
            else:
                chunk_entity_info += f" | Classes: {chunk['class_names']}"
        
        # Add the formatted chunk header
        context += f"\n--- CHUNK {i+1} ---\n{file_info} | {line_info}"
        
        # Add chunk-level entity info
        if chunk_entity_info:
            context += f"\n{chunk_entity_info}"
            
        # Add document-level entity info
        if doc_entity_info:
            context += f"\n{doc_entity_info}"
            
        # Add the content
        context += f"\n\n{chunk['content']}\n"
    
    # Create a more detailed prompt with specific instructions
    prompt = f"""You are an expert code assistant analyzing {codebase} codebase written in {language}.
I'll provide you with code chunks and a question. Your task is to:
1. Analyze each code chunk carefully
2. Identify the most relevant parts that answer the question
3. Explain the code's functionality and how it relates to the question
4. If appropriate, synthesize information across multiple chunks
5. Use specific line numbers and function names in your explanation
6. If the code implements a specific algorithm or pattern, identify and explain it

CODE CHUNKS:
{context}

QUESTION: {query}

ANSWER:"""
    
    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "HTTP-Referer": "https://code-rag-example.com",  # Replace with your actual domain
        "X-Title": "Code RAG"  # Optional site title for rankings
    }
    
    payload = {
        "model": "openai/gpt-3.5-turbo",  # Using a known working model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    # Make the API request
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)  # Using data=json.dumps() instead of json=
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract and return the generated answer
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: No response generated."
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

from weaviate.classes.query import Filter

from weaviate.collections.classes.filters import Filter  # correct import

def retrieve_chunks(
    query: str,
    top_k: int = 5,
    generate: bool = False,
    filter_chunks_flag: bool = False,
    use_summaries: bool = False,
    use_entity_filter: bool = True,
    codebase: str = DEFAULT_CODEBASE
):
    """v4: vector search + entity & summary filtering via static Filter.any_of()."""
    print(f"Processing query: {query}")

    # 1) Embed & client
    voyage_vectorizer = VoyageVectorizer()
    client = get_weaviate_client()
    class_name, summary_class_name = setup_weaviate_schema(client, codebase)
    query_embedding = voyage_vectorizer.get_embedding(query)

    # 2) Entity-based filters
    entities = extract_entities_from_query(query)
    filters = None
    if use_entity_filter and entities:
        print(f"Using entity-based filtering with: {entities}")
        entity_filters = []
        for e in entities:
            entity_filters += [
                Filter.by_property("chunk_function_names").contains_any([e]),
                Filter.by_property("document_function_names").contains_any([e]),
                Filter.by_property("chunk_class_names").contains_any([e]),
                Filter.by_property("document_class_names").contains_any([e]),
                Filter.by_property("filename").equal(e),
            ]
        # combine all entity filters with OR
        filters = Filter.any_of(entity_filters)                                # :contentReference[oaicite:0]{index=0}

    # 3) Summary-based filters
    if use_summaries:
        try:
            summary_coll = client.collections.get(summary_class_name)
            summary_resp = summary_coll.query.near_vector(
                near_vector=query_embedding,
                limit=top_k
            )
            relevant = [obj.properties["filename"] for obj in summary_resp.objects]
            if relevant:
                print(f"Relevant files from summaries: {relevant}")
                summary_filter = Filter.by_property("filename").contains_any(relevant)
                # combine with existing filters if present
                filters = Filter.any_of([*entity_filters, summary_filter]) if filters else summary_filter  # :contentReference[oaicite:1]{index=1}
                print("Applied summary-based file filtering")
        except Exception as e:
            print(f"Summary lookup failed ({e}), skipping summary filtering")
            use_summaries = False

    # 4) Main vector search
    chunk_coll = client.collections.get(class_name)
    resp = chunk_coll.query.near_vector(
        near_vector=query_embedding,
        filters=filters,
        limit=top_k
    )
    formatted = []
    for obj in resp.objects:
        p = obj.properties
        entry = {
            "filepath":        p.get("filepath"),
            "filename":        p.get("filename"),
            "relpath":         p.get("relpath"),
            "start_line":      p.get("start_line"),
            "end_line":        p.get("end_line"),
            "length":          p.get("length"),
            "language":        p.get("language"),
            "codebase":        codebase,
            "chunking_method": p.get("chunking_method"),
            "content":         p.get("content"),
        }
        # Add any list‐fields present
        for key in ("chunk_function_names","chunk_class_names",
                    "document_function_names","document_class_names"):
            if key in p:
                entry[key] = p[key]
        formatted.append(entry)

    print(f"Retrieved {len(formatted)} chunks")
    client.close()

    # 5) Optional post-filter & LLM‐based generation
    if filter_chunks_flag and formatted:
        formatted = filter_chunks(query, formatted, codebase)

    if generate and formatted:
        answer = generate_answer(query, formatted)
        return {"chunks": formatted, "answer": answer}

    return formatted


# def retrieve_chunks(
#     query,
#     top_k=5,
#     generate=False,
#     filter_chunks_flag=False,  # this still drives your LLM‐based filter step
#     use_summaries=False,
#     use_entity_filter=True,
#     codebase=DEFAULT_CODEBASE
# ):
#     """v4: vector search + optional Filter-based entity & summary narrowing."""
#     print(f"Processing query: {query}")

#     # 1) Embedding & client
#     voyage_vectorizer = VoyageVectorizer()
#     client            = get_weaviate_client()
#     class_name, summary_class_name = setup_weaviate_schema(client, codebase)
#     query_embedding   = voyage_vectorizer.get_embedding(query)

#     # 2) Entity‐based Filter
#     entities = extract_entities_from_query(query)
#     if not entities and use_entity_filter:
#         print("No specific code entities found, disabling entity‐based filtering")
#         use_entity_filter = False

#     filters = None
#     if use_entity_filter and entities:
#         # build OR across all entity fields
#         entity_filters = []
#         for e in entities:
#             entity_filters += [
#                 Filter.by_property("chunk_function_names").contains_any([e]),
#                 Filter.by_property("document_function_names").contains_any([e]),
#                 Filter.by_property("chunk_class_names").contains_any([e]),
#                 Filter.by_property("document_class_names").contains_any([e]),
#                 Filter.by_property("filename").equal(e),
#             ]
#         filters = Filter().or_(*entity_filters)
#         print(f"Using entity‐based filtering with entities: {entities}")

#     # 3) Summary‐based Filter
#     if use_summaries:
#         try:
#             summary_coll = client.collections.get(summary_class_name)
#             summary_resp = summary_coll.query.near_vector(
#                 near_vector=query_embedding,
#                 limit=top_k
#             )
#             relevant = [o.properties["filename"] for o in summary_resp.objects]
#             if relevant:
#                 print(f"Relevant files from summaries: {relevant}")
#                 summary_filter = Filter.by_property("filename").contains_any(relevant)
#                 filters = filters.or_(summary_filter) if filters else summary_filter
#                 print("Applied summary‐based file filtering")
#         except Exception as e:
#             print(f"Summary query failed ({e}), skipping summaries")
#             use_summaries = False

#     # 4) Main vector query on code‐chunk collection
#     chunk_coll = client.collections.get(class_name)
#     resp = chunk_coll.query.near_vector(
#         near_vector=query_embedding,
#         filters=filters,
#         limit=top_k
#     )  # returns all properties by default :contentReference[oaicite:0]{index=0}

#     formatted = []
#     for obj in resp.objects:
#         p = obj.properties
#         entry = {
#             "filepath": p.get("filepath"),
#             "filename": p.get("filename"),
#             "relpath":  p.get("relpath"),
#             "start_line": p.get("start_line"),
#             "end_line":   p.get("end_line"),
#             "length":     p.get("length"),
#             "language":   p.get("language"),
#             "codebase":   codebase,
#             "chunking_method": p.get("chunking_method"),
#             "content":    p.get("content"),
#         }
#         # arrays
#         for k in ("chunk_function_names","chunk_class_names",
#                   "document_function_names","document_class_names"):
#             if k in p:
#                 entry[k] = p[k]
#         formatted.append(entry)

#     print(f"Retrieved {len(formatted)} chunks")
#     client.close()

#     # 5) Optional LLM‐filter & generation
#     if filter_chunks_flag and formatted:
#         formatted = filter_chunks(query, formatted, codebase)
#     if generate and formatted:
#         answer = generate_answer(query, formatted)
#         return {"chunks": formatted, "answer": answer}

#     return formatted


# def retrieve_chunks(query, top_k=5, generate=False, filter_chunks_flag=False, use_summaries=False, use_entity_filter=True, codebase=DEFAULT_CODEBASE):
#     """Retrieve chunks from Weaviate based on the query with enhanced filtering.
    
#     Args:
#         query: The query string
#         top_k: Number of results to return
#         generate: Whether to generate an answer using LLM
#         filter_chunks_flag: Whether to filter chunks to remove irrelevant code
#         use_summaries: Whether to use file summaries to improve retrieval
#         use_entity_filter: Whether to use entity-based filtering (functions, classes)
#         codebase: Which codebase to use
#     """
#     print(f"Processing query: {query}")
    
#     # Extract specific code entities from the query
#     entities = extract_entities_from_query(query)
    
#     # If no entities were found and entity filtering is enabled, disable it
#     if not entities and use_entity_filter:
#         print("No specific code entities found, disabling entity-based filtering")
#         use_entity_filter = False
    
#     # Setup Weaviate with Voyage AI embeddings
#     voyage_vectorizer = VoyageVectorizer()
#     client = get_weaviate_client()
    
#     # Get the class names for the specified codebase
#     class_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["class_name"]
#     summary_class_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["summary_class_name"]
    
#     print(f"Using codebase configuration: {codebase}")
#     print(f"Retrieving from class: {class_name}")
    
#     # Get query embedding
#     query_embedding = voyage_vectorizer.get_embedding(query)
    
#     # Prepare where filter based on entity filtering
#     where_filter = None
#     if entities and use_entity_filter:
#         where_clauses = []
        
#         # Check for chunks containing functions
#         if entities:
#             where_clause = {
#                 "operator": "Or",
#                 "operands": []
#             }
            
#             # Add function name filters
#             for entity in entities:
#                 where_clause["operands"].append({
#                     "path": ["chunk_function_names"],
#                     "operator": "ContainsAny",
#                     "valueStringArray": [entity]
#                 })
                
#                 where_clause["operands"].append({
#                     "path": ["document_function_names"],
#                     "operator": "ContainsAny",
#                     "valueStringArray": [entity]
#                 })
                
#                 # Add class name filters
#                 where_clause["operands"].append({
#                     "path": ["chunk_class_names"],
#                     "operator": "ContainsAny",
#                     "valueStringArray": [entity]
#                 })
                
#                 where_clause["operands"].append({
#                     "path": ["document_class_names"],
#                     "operator": "ContainsAny",
#                     "valueStringArray": [entity]
#                 })
                
#                 # Add filename filter
#                 where_clause["operands"].append({
#                     "path": ["filename"],
#                     "operator": "Equal",
#                     "valueString": entity
#                 })
            
#             where_clauses.append(where_clause)
        
#         # Use the where filter if we have clauses
#         if where_clauses:
#             if len(where_clauses) > 1:
#                 where_filter = {
#                     "operator": "Or",
#                     "operands": where_clauses
#                 }
#             else:
#                 where_filter = where_clauses[0]
            
#             print(f"Using entity-based filtering with {len(entities)} entities")
    
#     # If we're using summaries, first find relevant file summaries
#     relevant_filenames = []
#     if use_summaries:
#         try:
#             # Query summary class to find relevant files
#             summary_results = client.query.get(
#                 summary_class_name, ["filename", "filepath", "relpath"]
#             ).with_near_vector(
#                 {"vector": query_embedding}
#             ).with_limit(top_k).do()
            
#             if "data" in summary_results and "Get" in summary_results["data"] and summary_results["data"]["Get"][summary_class_name]:
#                 summaries = summary_results["data"]["Get"][summary_class_name]
#                 for summary in summaries:
#                     relevant_filenames.append(summary["filename"])
                
#                 print(f"Found {len(relevant_filenames)} relevant files from summaries: {', '.join(relevant_filenames)}")
                
#                 # Create file filter
#                 if relevant_filenames and (not where_filter or not use_entity_filter):
#                     # Only use summary-based filtering if no entity filtering or if entity filtering didn't yield results
#                     where_filter = {
#                         "path": ["filename"],
#                         "operator": "ContainsAny",
#                         "valueStringArray": relevant_filenames
#                     }
#                     print("Using summary-based file filtering")
#                 elif relevant_filenames and where_filter and use_entity_filter:
#                     # Combine entity and summary filters
#                     combined_filter = {
#                         "operator": "Or",
#                         "operands": [
#                             where_filter,
#                             {
#                                 "path": ["filename"],
#                                 "operator": "ContainsAny",
#                                 "valueStringArray": relevant_filenames
#                             }
#                         ]
#                     }
#                     where_filter = combined_filter
#                     print("Using combined entity and summary-based filtering")
#         except Exception as e:
#             print(f"Error querying summary class: {e}")
#             print("Falling back to standard retrieval without summaries")
#             use_summaries = False
    
#     # Perform the vector search
#     try:
#         query_builder = client.query.get(
#             class_name, [
#                 "content", "filepath", "filename", "relpath",
#                 "start_line", "end_line", "length", "language",
#                 "chunking_method", "chunk_function_names",
#                 "chunk_class_names", "document_function_names",
#                 "document_class_names"
#             ]
#         ).with_near_vector(
#             {"vector": query_embedding}
#         ).with_limit(top_k)
        
#         # Add where filter if applicable
#         if where_filter:
#             query_builder = query_builder.with_where(where_filter)
        
#         # Execute the query
#         results = query_builder.do()
        
#         # Format the results
#         formatted_results = []
        
#         if "data" in results and "Get" in results["data"] and results["data"]["Get"][class_name]:
#             chunks = results["data"]["Get"][class_name]
            
#             for chunk in chunks:
#                 # Create a comprehensive result object with all available metadata
#                 result = {
#                     "filepath": chunk["filepath"],
#                     "filename": chunk["filename"],
#                     "relpath": chunk.get("relpath", ""),
#                     "start_line": chunk["start_line"],
#                     "end_line": chunk["end_line"],
#                     "length": chunk["length"],
#                     "language": chunk.get("language", ""),
#                     "codebase": codebase,
#                     "chunking_method": chunk.get("chunking_method", "unknown"),
#                     "content": chunk["content"]
#                 }
                
#                 # Add array properties
#                 if "chunk_function_names" in chunk:
#                     result["chunk_function_names"] = chunk["chunk_function_names"]
                
#                 if "chunk_class_names" in chunk:
#                     result["chunk_class_names"] = chunk["chunk_class_names"]
                
#                 if "document_function_names" in chunk:
#                     result["document_function_names"] = chunk["document_function_names"]
                
#                 if "document_class_names" in chunk:
#                     result["document_class_names"] = chunk["document_class_names"]
                
#                 formatted_results.append(result)
            
#             print(f"Retrieved {len(formatted_results)} chunks")
#         else:
#             print("No results found")
    
#     except Exception as e:
#         print(f"Error retrieving chunks: {e}")
#         return []
    
#     # If filter flag is set, filter chunks using LLM
#     if filter_chunks_flag and formatted_results:
#         formatted_results = filter_chunks(query, formatted_results, codebase)
    
#     # Prepare the return object
#     result_object = {"chunks": formatted_results}
    
#     # If generate flag is set, generate an answer using the retrieved chunks
#     if generate and formatted_results:
#         print("generate flag is set")
#         answer = generate_answer(query, formatted_results)
#         result_object["answer"] = answer
#         # Return the full result object with both chunks and answer when generate is true
#         return result_object
    
#     # Return just the formatted results when generate is false (for compatibility with retrieval-perf.py)
#     return formatted_results

# def reset_db(codebase=DEFAULT_CODEBASE):
#     """v4: delete & recreate the two collections."""
#     client = get_weaviate_client()
#     cfg = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])
#     chunk_cls   = cfg["class_name"]
#     summary_cls = cfg["summary_class_name"]

#     # delete if exists (ignore errors)
#     for cls in (chunk_cls, summary_cls):
#         try:
#             client.collections.delete(cls)
#             print(f"Deleted collection: {cls}")
#         except Exception:
#             pass

#     # now recreate schema
#     setup_weaviate_schema(client, codebase)
#     client.close()
#     print("Reset complete.")
#def reset_db(codebase=DEFAULT_CODEBASE):
#    """Reset the Weaviate database by deleting and recreating classes."""
#    client = get_weaviate_client()
#    voyage_vectorizer = VoyageVectorizer()
#    
#    # Get the class names for the specified codebase
#    class_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["class_name"]
#    summary_class_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["summary_class_name"]
#    print(f"Using codebase configuration: {codebase}")
#    print(f"Resetting classes: {class_name} and {summary_class_name}")
#    
#    # Delete existing classes if they exist
#    try:
#        client.schema.delete_class(class_name)
#        print(f"Successfully deleted class '{class_name}'")
#    except Exception as e:
#        print(f"Class '{class_name}' doesn't exist or could not be deleted: {e}")
#    
#    try:
#        client.schema.delete_class(summary_class_name)
#        print(f"Successfully deleted class '{summary_class_name}'")
#    except Exception as e:
#        print(f"Class '{summary_class_name}' doesn't exist or could not be deleted: {e}")
#    
#    # Recreate empty classes with proper schema
#    setup_weaviate_schema(client, voyage_vectorizer, codebase)
#    print(f"Created new empty classes '{class_name}' and '{summary_class_name}'")

def generate_summary(code):
    """Generate a summary using OpenRouter LLM based on the given code."""
    # Check for API key in environment
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Warning: OPENROUTER_API_KEY environment variable not set.")
        print("Please set your OpenRouter API key in the .env file: OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Create the prompt
    prompt = f"""You are an expert code assistant that helps explain code.
I'll provide you with code from a file and you will generate a summary of the code.
Please provide a clear, concise explanation of the code.

CODE:
{code}

SUMMARY:"""
    
    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "HTTP-Referer": "https://code-rag-example.com",  # Replace with your actual domain
        "X-Title": "Code RAG"  # Optional site title for rankings
    }
    
    payload = {
        "model": "openai/gpt-3.5-turbo",  # Using a known working model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    # Make the API request
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)  # Using data=json.dumps() instead of json=
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract and return the generated summary
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: No response generated."
    
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Code RAG - Weaviate-based code retrieval system")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add codebase argument to parent parser so it's available to all subcommands
    parser.add_argument("--codebase", choices=["xv6", "llama_index"], default=DEFAULT_CODEBASE,
                        help="Specify which codebase to use (affects chunking parameters and collection)")
    
    # Chunker command
    chunker_parser = subparsers.add_parser("chunker", help="Process source tree and save chunks to JSON")
    chunker_parser.add_argument("source_dir", help="Path to source directory")
    chunker_parser.add_argument("output_file", help="Output JSON file path")
    
    # Indexer command
    indexer_parser = subparsers.add_parser("indexer", help="Index chunks from JSON file to Weaviate")
    indexer_parser.add_argument("chunk_file", help="Input JSON file with chunks")
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve chunks based on query")
    query_group = retrieve_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("query", nargs='?', help="Query string for retrieval")
    query_group.add_argument("-q", "--query-file", help="Path to a text file containing the query")
    retrieve_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return")
    retrieve_parser.add_argument("-g", "--generate", action="store_true", help="Generate an answer using LLM")
    retrieve_parser.add_argument("-f", "--filter", action="store_true", help="Filter retrieved chunks to remove irrelevant code")
    retrieve_parser.add_argument("-s", "--summary", action="store_true", help="Use file summaries to improve retrieval")
    retrieve_parser.add_argument("-e", "--entity-filter", dest="entity_filter", action="store_true", default=True,
                                help="Use entity-based filtering (functions, classes)")
    retrieve_parser.add_argument("--no-entity-filter", dest="entity_filter", action="store_false",
                                help="Disable entity-based filtering")
    retrieve_parser.add_argument("-o", "--output", help="Path to save retrieved chunks as JSON file")
    
    # Reset command
    subparsers.add_parser("resetdb", help="Reset the Weaviate database")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "chunker":
        chunk_source_tree(args.source_dir, args.output_file, args.codebase)
    
    elif args.command == "indexer":
        index_chunks(args.chunk_file, args.codebase)
    elif args.command == "retrieve":
        # Get the query from either the command line or a file
        query = args.query
        if args.query_file:
            try:
                with open(args.query_file, 'r', encoding='utf-8') as f:
                    query = f.read().strip()
                print(f"Loaded query from file: {args.query_file}")
                print(f"Query: {query}")
            except Exception as e:
                print(f"Error loading query from file {args.query_file}: {e}")
                sys.exit(1)
        
        results = retrieve_chunks(query, args.top_k, args.generate, args.filter, args.summary, args.entity_filter, args.codebase)
        
        # Save results to file if output path is provided
        if args.output:
            try:
                # If generate flag is set, save only the answer as text
                if args.generate and isinstance(results, dict) and 'answer' in results:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(results['answer'])
                    print(f"Generated answer saved to {args.output}")
                # Otherwise, save the chunks as JSON (original behavior)
                else:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        # Check if results is a dict with 'chunks' key (new format) or a list (old format)
                        if isinstance(results, dict) and 'chunks' in results:
                            json.dump(results, f, indent=2)
                        else:
                            json.dump(results, f, indent=2)
                    print(f"Retrieved chunks saved to {args.output}")
            except Exception as e:
                print(f"Error saving results to {args.output}: {e}")
        
        # Always print results to console
        print(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    
    elif args.command == "resetdb":
        reset_db(args.codebase)

if __name__ == "__main__":
    main()
