#!/usr/bin/env python3
"""
Code RAG - Retrieval Augmented Generation for code search using ChromaDB and OpenAI embeddings.

This script provides functionality for chunking source code, indexing those chunks in a
ChromaDB vector database, and retrieving relevant code snippets based on semantic similarity.

Usage:
    python code-rag.py chunker <path-to-source-tree> <output-json-file> [--language/-l <language>]
    python code-rag.py indexer <chunk-list-json-file> [--db-path <path>]
    python code-rag.py retrieve <user prompt for vector similarity search> [--db-path <path>]
    python code-rag.py resetdb [--db-path <path>]

Example:
    python code-rag.py chunker ../xv6-riscv chunk-list.json
    python code-rag.py chunker ../xv6-riscv chunk-list.json --language c
    python code-rag.py chunker ../python-project chunk-list.json -l python
    python code-rag.py indexer chunk-list.json
    python code-rag.py indexer chunk-list.json --db-path custom_db
    python code-rag.py retrieve "how does file system initialization work?"
    python code-rag.py retrieve "how does file system initialization work?" --db-path custom_db
    python code-rag.py resetdb
    python code-rag.py resetdb --db-path custom_db
"""

import os
import sys
import json
import argparse
from pathlib import Path
import importlib.util
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException
import openai

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
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure you have the required packages installed:")
    print("pip install llama-index tree-sitter tree-sitter-languages chromadb openai")
    sys.exit(1)

# Constants
DEFAULT_DB_DIRECTORY = "code_chunks_db"
COLLECTION_NAME = "code_chunks"

def create_openai_ef():
    """Create OpenAI embedding function for ChromaDB."""
    # Check for API key in environment
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
        
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        #model_name="text-embedding-3-small"
        model_name="text-embedding-ada-002"
    )

def find_source_files(directory, language="c"):
    """Find all source files in directory and subdirectories based on language."""
    source_files = []
    extensions = ['.c', '.h'] if language == "c" else ['.py']
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                source_files.append(os.path.join(root, file))
    return source_files

def process_file(file_path, base_dir, language="c"):
    """Process a source file and return its chunks with metadata."""
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        
        # Create a Document from the code with file path information
        document = Document(
            text=code,
            metadata={
                "filepath": file_path,
                "base_dir": base_dir
            }
        )
        
        # Configure the CodeSplitter
        code_splitter = CodeSplitter(
            language=language,
            chunk_lines=60,  # Adjust as needed
            chunk_lines_overlap=5,  # Adjust as needed
            max_chars=2048,  # Adjust as needed
        )
        
        # Split the document into nodes with enhanced metadata
        nodes = code_splitter.get_nodes_from_documents([document])
        
        # Convert nodes to the required format
        chunks = []
        for node in nodes:
            chunk = {
                "filepath": node.metadata.get("filepath", ""),
                "filename": node.metadata.get("filename", ""),
                "relpath": node.metadata.get("relpath", ""),
                "start_line": node.metadata.get("start_line", 0),
                "end_line": node.metadata.get("end_line", 0),
                "length": node.metadata.get("end_line", 0) - node.metadata.get("start_line", 0) + 1,
                "content": node.text
            }
            chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def chunk_source_tree(source_dir, output_file, language="c"):
    """Process all files in the source directory and save chunks to a JSON file."""
    # Check if the directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        sys.exit(1)
    
    # Get the absolute path
    abs_dir_path = os.path.abspath(source_dir)
    
    # Find all source files
    source_files = find_source_files(abs_dir_path, language)
    print(f"Found {len(source_files)} source files in {abs_dir_path}")
    
    # Process each file and collect chunks
    all_chunks = []
    for file_path in source_files:
        print(f"Processing {file_path}")
        chunks = process_file(file_path, abs_dir_path, language)
        all_chunks.extend(chunks)
        print(f"  - Generated {len(chunks)} chunks")
    
    # Save the chunks to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\nSuccessfully processed {len(source_files)} files.")
    print(f"Generated {len(all_chunks)} chunks.")
    print(f"Results saved to {output_file}")

def get_chroma_client(db_path=DEFAULT_DB_DIRECTORY):
    """Get or create a ChromaDB client."""
    return chromadb.PersistentClient(path=db_path)

def get_collection(client, embedding_function=None):
    """Get or create the collection for code chunks."""
    try:
        # Try to get existing collection
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    except InvalidCollectionException:
        # Collection doesn't exist, create it
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

def index_chunks(chunk_file, db_path=DEFAULT_DB_DIRECTORY):
    """Index chunks from a JSON file into ChromaDB with OpenAI embeddings."""
    # Load chunks from JSON
    try:
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error loading chunks from {chunk_file}: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")
    
    # Setup ChromaDB with OpenAI embeddings
    embedding_function = create_openai_ef()
    client = get_chroma_client(db_path)
    collection = get_collection(client, embedding_function)
    
    # Prepare data for batch upload
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        
        # Extract metadata (excluding content to avoid duplication)
        metadata = {
            "filepath": chunk["filepath"],
            "filename": chunk["filename"],
            "relpath": chunk["relpath"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "length": chunk["length"]
        }
        metadatas.append(metadata)
    
    # Add documents to collection in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"Indexing chunks {i} to {end_idx-1}...")
        
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    
    count = collection.count()
    print(f"Successfully indexed {count} chunks in ChromaDB")

def retrieve_chunks(query, top_k=5, db_path=DEFAULT_DB_DIRECTORY):
    """Retrieve chunks from ChromaDB based on the query."""
    # Setup ChromaDB with OpenAI embeddings
    embedding_function = create_openai_ef()
    client = get_chroma_client(db_path)
    
    try:
        collection = get_collection(client, embedding_function)
    except InvalidCollectionException:
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
        print("Make sure you have indexed chunks with 'python code-rag.py indexer <chunk-file>' before trying to retrieve.")
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        sys.exit(1)
    
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Format results to match the required output format
    formatted_results = []
    
    if results and results["metadatas"] and results["documents"]:
        for metadata, document in zip(results["metadatas"][0], results["documents"][0]):
            result = {
                "filepath": metadata["filepath"],
                "filename": metadata["filename"],
                "relpath": metadata["relpath"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "length": metadata["length"],
                "content": document
            }
            formatted_results.append(result)
    
    return formatted_results

def reset_db(db_path=DEFAULT_DB_DIRECTORY):
    """Reset the ChromaDB database by deleting and recreating it."""
    client = get_chroma_client(db_path)
    
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Successfully deleted collection '{COLLECTION_NAME}'")
    except (InvalidCollectionException, ValueError) as e:
        print(f"Collection '{COLLECTION_NAME}' doesn't exist or already deleted")
    
    # Recreate an empty collection
    embedding_function = create_openai_ef()
    get_collection(client, embedding_function)
    print(f"Created a new empty collection '{COLLECTION_NAME}'")

def main():
    parser = argparse.ArgumentParser(description="Code RAG - ChromaDB-based code retrieval system")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Chunker command
    chunker_parser = subparsers.add_parser("chunker", help="Process source tree and save chunks to JSON")
    chunker_parser.add_argument("source_dir", help="Path to source directory")
    chunker_parser.add_argument("output_file", help="Output JSON file path")
    chunker_parser.add_argument("-l", "--language", choices=["c", "python"], default="c",
                               help="Source code language (default: c)")
    
    # Indexer command
    indexer_parser = subparsers.add_parser("indexer", help="Index chunks from JSON file to ChromaDB")
    indexer_parser.add_argument("chunk_file", help="Input JSON file with chunks")
    indexer_parser.add_argument("--db-path", default=DEFAULT_DB_DIRECTORY,
                               help=f"Path to ChromaDB database (default: {DEFAULT_DB_DIRECTORY})")
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve chunks based on query")
    retrieve_parser.add_argument("query", help="Query string for retrieval")
    retrieve_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return")
    retrieve_parser.add_argument("--db-path", default=DEFAULT_DB_DIRECTORY,
                               help=f"Path to ChromaDB database (default: {DEFAULT_DB_DIRECTORY})")
    
    # Reset command
    resetdb_parser = subparsers.add_parser("resetdb", help="Reset the ChromaDB database")
    resetdb_parser.add_argument("--db-path", default=DEFAULT_DB_DIRECTORY,
                              help=f"Path to ChromaDB database (default: {DEFAULT_DB_DIRECTORY})")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "chunker":
        chunk_source_tree(args.source_dir, args.output_file, args.language)
    
    elif args.command == "indexer":
        index_chunks(args.chunk_file, args.db_path)
    
    elif args.command == "retrieve":
        results = retrieve_chunks(args.query, args.top_k, args.db_path)
        print(json.dumps(results, indent=2))
    
    elif args.command == "resetdb":
        reset_db(args.db_path)

if __name__ == "__main__":
    main()