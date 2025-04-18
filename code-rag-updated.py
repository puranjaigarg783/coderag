#!/usr/bin/env python3
"""
Code RAG - Retrieval Augmented Generation for code search using ChromaDB and Voyage AI embeddings.

This script provides functionality for chunking source code, indexing those chunks in a
ChromaDB vector database, and retrieving relevant code snippets based on semantic similarity.

Usage:
    python code-rag-updated.py --codebase <codebase> chunker <path-to-source-tree> <output-json-file>
    python code-rag-updated.py --codebase <codebase> indexer <chunk-list-json-file>
    python code-rag-updated.py --codebase <codebase> retrieve <user prompt for vector similarity search> [-k <num-results>] [-g] [-f] [-s] [-e|--no-entity-filter] [-o <output-file>]
    python code-rag-updated.py --codebase <codebase> retrieve -q <query-file.txt> [-k <num-results>] [-g] [-f] [-s] [-e|--no-entity-filter] [-o <output-file>]
    python code-rag-updated.py --codebase <codebase> resetdb

Example:
    # For xv6 codebase
    python code-rag-updated.py --codebase xv6 chunker ../xv6-riscv xv6-chunks.json
    python code-rag-updated.py --codebase xv6 indexer xv6-chunks.json
    python code-rag-updated.py --codebase xv6 retrieve "how does file system initialization work?"
    python code-rag-updated.py --codebase xv6 retrieve -f "how does file system initialization work?"
    python code-rag-updated.py --codebase xv6 retrieve -s "how does file system initialization work?"
    python code-rag-updated.py --codebase xv6 retrieve -s -e "how does file system initialization work?"  # Use both summary and entity filtering
    python code-rag-updated.py --codebase xv6 retrieve -s --no-entity-filter "how does file system initialization work?"  # Use only summary filtering
    python code-rag-updated.py --codebase xv6 retrieve "how does file system initialization work?" -o results.json  # Save results to JSON file
    python code-rag-updated.py --codebase xv6 retrieve -q query.txt -o results.json  # Load query from file and save results to JSON
    python code-rag-updated.py --codebase xv6 resetdb
    
    # For llama_index codebase
    python code-rag-updated.py --codebase llama_index chunker ./data/llama_index llama-chunks.json
    python code-rag-updated.py --codebase llama_index indexer llama-chunks.json
    python code-rag-updated.py --codebase llama_index retrieve "how does the embedding API work?"
    python code-rag-updated.py --codebase llama_index retrieve -g "explain the retriever implementation"
    python code-rag-updated.py --codebase llama_index retrieve -g -f "explain the retriever implementation"
    python code-rag-updated.py --codebase llama_index retrieve -g -s "explain the retriever implementation"
    python code-rag-updated.py --codebase llama_index retrieve -g -s -e "explain the retriever implementation"
    python code-rag-updated.py --codebase llama_index retrieve -g "explain the retriever implementation" -o retriever_answer.txt  # Save only the answer to text file
    python code-rag-updated.py --codebase llama_index retrieve "explain the retriever implementation" -o retriever_chunks.json  # Save chunks to JSON file

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
import argparse
import requests
from pathlib import Path
import importlib.util
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException
from chromadb import Documents, EmbeddingFunction, Embeddings
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
    print("pip install llama-index tree-sitter tree-sitter-languages chromadb python-dotenv")
    print("pip install llama-index-embeddings-voyageai")
    sys.exit(1)

# Constants
DB_DIRECTORY = "code_chunks_db"

# Codebase-specific configurations
CODEBASE_CONFIGS = {
    "xv6": {
        "collection_name": "xv6_code_chunks",
        "chunk_lines": 60,  # Reduced from 60 for more focused chunks
        "chunk_lines_overlap": 5,  # Increased from 5 for better context preservation
        "max_chars": 2048  # Reduced from 2048 for more focused chunks
    },
    "llama_index": {
        "collection_name": "llama_index_code_chunks",
        "chunk_lines": 500,  # Reduced from 500 for more granular retrieval
        "chunk_lines_overlap": 1,  # Increased from 1 for better context preservation
        "max_chars": 16384  # Reduced from 16384 for more focused chunks
    }
}

# Default to xv6 if not specified
DEFAULT_CODEBASE = "xv6"

class VoyageEmbeddingFunction(EmbeddingFunction):
    """Voyage AI embedding function for ChromaDB that follows the required interface."""
    
    def __init__(self, voyage_embedding):
        self.voyage_embedding = voyage_embedding
        
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            embedding = self.voyage_embedding.get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings

def create_voyage_ef():
    """Create Voyage AI embedding function for ChromaDB."""
    # Check for API key in environment
    if "VOYAGE_API_KEY" not in os.environ:
        print("Warning: VOYAGE_API_KEY environment variable not set.")
        print("Please set your Voyage API key in the .env file: VOYAGE_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Create a VoyageEmbedding instance
    voyage_embedding = VoyageEmbedding(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
        model_name="voyage-3"
    )
    
    #print("created voyage AI embedding function for chroma db")
    return VoyageEmbeddingFunction(voyage_embedding)
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

def get_chroma_client(existing=False):
    """Get or create a ChromaDB client."""
    #print("got chroma client")
    return chromadb.PersistentClient(path=DB_DIRECTORY)

def get_collection(client, embedding_function=None, codebase=DEFAULT_CODEBASE):
    """Get or create the collection for code chunks."""
    # Get the collection name for the specified codebase
    collection_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["collection_name"]
    
    try:
        # Try to get existing collection
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except InvalidCollectionException:
        # Collection doesn't exist, create it
        print(f"Creating new collection: {collection_name}")
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

def index_chunks(chunk_file, codebase=DEFAULT_CODEBASE):
    """Index chunks and summaries into ChromaDB with Voyage AI embeddings."""
    # Load chunks from JSON
    try:
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error loading chunks from {chunk_file}: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")
    print(f"Using codebase configuration: {codebase}")
    
    # Setup ChromaDB with Voyage AI embeddings
    embedding_function = create_voyage_ef()
    client = get_chroma_client()
    collection = get_collection(client, embedding_function, codebase)
    
    # Load summaries from code_summaries.json
    try:
        with open("code_summaries.json", 'r') as f:
            summaries = json.load(f)
        print(f"Loaded {len(summaries)} summaries from code_summaries.json")
    except Exception as e:
        print(f"Error loading summaries from code_summaries.json: {e}")
        print("Will proceed without summaries")
        summaries = []
    
    # Create a new ChromaDB collection for summaries
    summary_collection_name = f"{CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])['collection_name']}_summaries"
    try:
        summary_collection = client.get_collection(
            name=summary_collection_name,
            embedding_function=embedding_function
        )
    except InvalidCollectionException:
        print(f"Creating new collection for summaries: {summary_collection_name}")
        summary_collection = client.create_collection(
            name=summary_collection_name,
            embedding_function=embedding_function
        )
    
    # Add each summary as a single document to the summary collection
    if summaries:
        print(f"Indexing {len(summaries)} file summaries...")
        for i, summary in enumerate(summaries):
            summary_id = f"summary_{i}"
            summary_document = summary["summary"]
            summary_metadata = {
                "filepath": summary["path"],
                "filename": summary["filename"],
                "relpath": summary["relpath"],
            }
            try:
                summary_collection.add(
                    ids=[summary_id],
                    documents=[summary_document],
                    metadatas=[summary_metadata]
                )
            except Exception as e:
                print(f"Error indexing summary {i}: {e}")
        
        print(f"Successfully indexed {len(summaries)} summaries in ChromaDB collection '{summary_collection_name}'")
    
    # Prepare data for batch upload of chunks
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        
        # Extract enhanced metadata (excluding content to avoid duplication)
        metadata = {
            "filepath": chunk["filepath"],
            "filename": chunk["filename"],
            "relpath": chunk.get("relpath", ""),
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "length": chunk["length"],
            "language": chunk.get("language", ""),
            "codebase": chunk.get("codebase", codebase),
            "chunking_method": chunk.get("chunking_method", "unknown")
        }
        
        # Add chunk-specific function and class names
        if "chunk_function_names" in chunk and chunk["chunk_function_names"]:
            metadata["chunk_function_names"] = ",".join(chunk["chunk_function_names"])
            # Add individual function presence flags for easier filtering
            for func_name in chunk["chunk_function_names"]:
                # Use a safe key name (replace invalid chars)
                safe_key = f"has_func_{func_name}"[:63].replace(".", "_").replace("-", "_")
                metadata[safe_key] = True
        
        if "chunk_class_names" in chunk and chunk["chunk_class_names"]:
            metadata["chunk_class_names"] = ",".join(chunk["chunk_class_names"])
            # Add individual class presence flags
            for class_name in chunk["chunk_class_names"]:
                safe_key = f"has_class_{class_name}"[:63].replace(".", "_").replace("-", "_")
                metadata[safe_key] = True
        
        # Add document-level function and class names
        if "document_function_names" in chunk and chunk["document_function_names"]:
            metadata["document_function_names"] = ",".join(chunk["document_function_names"])
            # Add document-level function flags with a different prefix
            for func_name in chunk["document_function_names"]:
                safe_key = f"doc_has_func_{func_name}"[:63].replace(".", "_").replace("-", "_")
                metadata[safe_key] = True
        
        if "document_class_names" in chunk and chunk["document_class_names"]:
            metadata["document_class_names"] = ",".join(chunk["document_class_names"])
            # Add document-level class flags
            for class_name in chunk["document_class_names"]:
                safe_key = f"doc_has_class_{class_name}"[:63].replace(".", "_").replace("-", "_")
                metadata[safe_key] = True
                
        # For backward compatibility with existing code
        if "function_names" in chunk and chunk["function_names"]:
            metadata["function_names"] = ",".join(chunk["function_names"])
            
        if "class_names" in chunk and chunk["class_names"]:
            metadata["class_names"] = ",".join(chunk["class_names"])
        
        metadatas.append(metadata)
    
    print(f"Prepared {len(ids)} chunks with enhanced metadata for indexing")
    
    # Add documents to collection in batches
    batch_size = 250
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"Indexing chunks {i} to {end_idx-1}...")
        
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    
    count = collection.count()
    print(f"Successfully indexed {count} chunks in ChromaDB collection '{collection.name}'")

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
def retrieve_chunks(query, top_k=5, generate=False, filter_chunks_flag=False, use_summaries=False, use_entity_filter=True, codebase=DEFAULT_CODEBASE):
    """Retrieve chunks from ChromaDB based on the query with enhanced metadata filtering and optional summary-based filtering.
    
    Args:
        query: The query string
        top_k: Number of results to return
        generate: Whether to generate an answer using LLM
        filter_chunks_flag: Whether to filter chunks to remove irrelevant code
        use_summaries: Whether to use file summaries to improve retrieval
        use_entity_filter: Whether to use entity-based filtering (functions, classes)
        codebase: Which codebase to use
    """
    print(f"Processing query: {query}")
    
    # Extract specific code entities from the query
    entities = extract_entities_from_query(query)
    print(f"DEBUG: Extracted entities: {entities}")
    
    # If no entities were found and entity filtering is enabled, disable it
    if not entities and use_entity_filter:
        print("No specific code entities found, disabling entity-based filtering")
        use_entity_filter = False
    
    # Setup ChromaDB with Voyage AI embeddings
    embedding_function = create_voyage_ef()
    client = get_chroma_client()
    
    # Get the collection name for the specified codebase
    collection_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["collection_name"]
    summary_collection_name = f"{collection_name}_summaries"
    print(f"Using codebase configuration: {codebase}")
    print(f"Retrieving from collection: {collection_name}")
    
    try:
        collection = get_collection(client, embedding_function, codebase)
        
        # If using summaries, also get the summary collection
        if use_summaries:
            try:
                summary_collection = client.get_collection(
                    name=summary_collection_name,
                    embedding_function=embedding_function
                )
                print(f"Using summary collection: {summary_collection_name}")
            except InvalidCollectionException:
                print(f"Warning: Summary collection '{summary_collection_name}' does not exist.")
                print(f"Falling back to standard retrieval without summaries.")
                use_summaries = False
    except InvalidCollectionException:
        print(f"Error: Collection '{collection_name}' does not exist.")
        print(f"Make sure you have indexed chunks for the '{codebase}' codebase before trying to retrieve.")
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        sys.exit(1)
    
    # Prepare metadata filters if entities were found
    where_filter = None
    if entities:
        # Create a filter to match function names or class names
        where_clauses = []
        # Use boolean flags for exact function matching
        function_clauses = []
        for entity in entities:
            # Create safe key names
            safe_entity = entity.replace(".", "_").replace("-", "_")
            
            # Check for chunk-level function (has_func_X)
            chunk_func_key = f"has_func_{safe_entity}"
            function_clauses.append({chunk_func_key: True})
            
            # Check for document-level function (doc_has_func_X)
            doc_func_key = f"doc_has_func_{safe_entity}"
            function_clauses.append({doc_func_key: True})
        
        print(f"DEBUG: Function clauses: {function_clauses}")
        
        # Only use $or if we have multiple clauses, otherwise use the single clause directly
        if len(function_clauses) > 1:
            where_clauses.append({"$or": function_clauses})
        elif len(function_clauses) == 1:
            # If there's only one function clause, add it directly without $or
            where_clauses.append(function_clauses[0])
        
        # Use boolean flags for exact class matching
        class_clauses = []
        for entity in entities:
            # Create safe key names
            safe_entity = entity.replace(".", "_").replace("-", "_")
            
            # Check for chunk-level class (has_class_X)
            chunk_class_key = f"has_class_{safe_entity}"
            class_clauses.append({chunk_class_key: True})
            
            # Check for document-level class (doc_has_class_X)
            doc_class_key = f"doc_has_class_{safe_entity}"
            class_clauses.append({doc_class_key: True})
            
        print(f"DEBUG: Class clauses: {class_clauses}")
        
        # Only use $or if we have multiple clauses, otherwise use the single clause directly
        if len(class_clauses) > 1:
            where_clauses.append({"$or": class_clauses})
        elif len(class_clauses) == 1:
            # If there's only one class clause, add it directly without $or
            where_clauses.append(class_clauses[0])
            
        # Check for filenames using $eq for exact matching
        file_clauses = []
        for entity in entities:
            # For filename, we can use $eq for exact matching
            file_clauses.append({"filename": {"$eq": entity}})
            
        print(f"DEBUG: File clauses: {file_clauses}")
        
        # Only use $or if we have multiple clauses, otherwise use the single clause directly
        if len(file_clauses) > 1:
            where_clauses.append({"$or": file_clauses})
        elif len(file_clauses) == 1:
            # If there's only one file clause, add it directly without $or
            where_clauses.append(file_clauses[0])
        
        # Combine all clauses with OR, but only if we have multiple clauses
        if len(where_clauses) > 1:
            where_filter = {"$or": where_clauses}
            print(f"Using metadata filter with OR: {where_filter}")
        elif len(where_clauses) == 1:
            # If there's only one where clause, use it directly without $or
            where_filter = where_clauses[0]
            print(f"Using metadata filter directly: {where_filter}")
    
    # If using summaries, first fetch related files from summary collection
    summary_filter = None
    related_filenames = set()
    if use_summaries:
        # 1. Fetch related files from summary collection
        summary_results = summary_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract filenames from summary results
        if summary_results and summary_results["metadatas"]:
            for metadata in summary_results["metadatas"][0]:
                related_filenames.add(metadata["filename"])
            
            if related_filenames:
                summary_filter = {"filename": {"$in": list(related_filenames)}}
                print(f"Found {len(related_filenames)} relevant files from summaries: {', '.join(related_filenames)}")
    
    # Determine which filters to apply
    combined_filter = None
    
    # Case 1: Both filters active
    if use_summaries and summary_filter and use_entity_filter and where_filter:
        combined_filter = {"$and": [summary_filter, where_filter]}
        print(f"Querying with combined summary and entity filtering")
    
    # Case 2: Only summary filter active
    elif use_summaries and summary_filter:
        combined_filter = summary_filter
        print(f"Querying with summary-based file filtering only")
    
    # Case 3: Only entity filter active
    elif use_entity_filter and where_filter:
        combined_filter = where_filter
        print(f"Querying with entity-based filtering only")
    
    # Case 4: No filters active
    else:
        print("No filtering applied - using semantic search only")
    
    # Query the collection with the appropriate filtering
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=combined_filter
    )

    # Format results to match the required output format
    formatted_results = []
    
    if results and results["metadatas"] and results["documents"]:
        for metadata, document in zip(results["metadatas"][0], results["documents"][0]):
            # Create a comprehensive result object with all available metadata
            result = {
                "filepath": metadata["filepath"],
                "filename": metadata["filename"],
                "relpath": metadata.get("relpath", ""),
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "length": metadata["length"],
                "language": metadata.get("language", ""),
                "codebase": metadata.get("codebase", codebase),
                "chunking_method": metadata.get("chunking_method", "unknown"),
                "content": document
            }
            
            # Add chunk-level function and class names
            if "chunk_function_names" in metadata:
                result["chunk_function_names"] = metadata["chunk_function_names"].split(",") if metadata["chunk_function_names"] else []
            
            if "chunk_class_names" in metadata:
                result["chunk_class_names"] = metadata["chunk_class_names"].split(",") if metadata["chunk_class_names"] else []
                
            # Add document-level function and class names
            if "document_function_names" in metadata:
                result["document_function_names"] = metadata["document_function_names"].split(",") if metadata["document_function_names"] else []
            
            if "document_class_names" in metadata:
                result["document_class_names"] = metadata["document_class_names"].split(",") if metadata["document_class_names"] else []
            
            # For backward compatibility
            if "function_names" in metadata:
                result["function_names"] = metadata["function_names"].split(",") if metadata["function_names"] else []
            
            if "class_names" in metadata:
                result["class_names"] = metadata["class_names"].split(",") if metadata["class_names"] else []
                
            formatted_results.append(result)
        
        print(f"Retrieved {len(formatted_results)} chunks")
    
    # If filter flag is set, filter chunks using LLM
    if filter_chunks_flag and formatted_results:
        formatted_results = filter_chunks(query, formatted_results, codebase)
    
    # Prepare the return object
    result_object = {"chunks": formatted_results}
    
    # If generate flag is set, generate an answer using the retrieved chunks
    if generate and formatted_results:
        print("generate flag is set")
        answer = generate_answer(query, formatted_results)
        result_object["answer"] = answer
        # Return the full result object with both chunks and answer when generate is true
        return result_object
    
    # Return just the formatted results when generate is false (for compatibility with retrieval-perf.py)
    return formatted_results

def reset_db(codebase=DEFAULT_CODEBASE):
    """Reset the ChromaDB database by deleting and recreating it."""
    client = get_chroma_client()
    
    # Get the collection name for the specified codebase
    collection_name = CODEBASE_CONFIGS.get(codebase, CODEBASE_CONFIGS[DEFAULT_CODEBASE])["collection_name"]
    summary_collection_name = f"{collection_name}_summaries"
    print(f"Using codebase configuration: {codebase}")
    print(f"Resetting collections: {collection_name} and {summary_collection_name}")
    
    try:
        client.delete_collection(collection_name)
        print(f"Successfully deleted collection '{collection_name}'")
    except (InvalidCollectionException, ValueError) as e:
        print(f"Collection '{collection_name}' doesn't exist or already deleted")
    
    try:
        client.delete_collection(summary_collection_name)
        print(f"Successfully deleted collection '{summary_collection_name}'")
    except (InvalidCollectionException, ValueError) as e:
        print(f"Collection '{summary_collection_name}' doesn't exist or already deleted")
    
    # Recreate empty collections
    embedding_function = create_voyage_ef()
    get_collection(client, embedding_function, codebase)
    print(f"Created a new empty collection '{collection_name}'")
    
    # Create summary collection
    client.create_collection(
        name=summary_collection_name,
        embedding_function=embedding_function
    )
    print(f"Created a new empty collection '{summary_collection_name}'")

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
    parser = argparse.ArgumentParser(description="Code RAG - ChromaDB-based code retrieval system")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add codebase argument to parent parser so it's available to all subcommands
    parser.add_argument("--codebase", choices=["xv6", "llama_index"], default=DEFAULT_CODEBASE,
                        help="Specify which codebase to use (affects chunking parameters and collection)")
    
    # Chunker command
    chunker_parser = subparsers.add_parser("chunker", help="Process source tree and save chunks to JSON")
    chunker_parser.add_argument("source_dir", help="Path to source directory")
    chunker_parser.add_argument("output_file", help="Output JSON file path")
    
    # Indexer command
    indexer_parser = subparsers.add_parser("indexer", help="Index chunks from JSON file to ChromaDB")
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
    subparsers.add_parser("resetdb", help="Reset the ChromaDB database")
    
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
