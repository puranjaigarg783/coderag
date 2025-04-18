#!/usr/bin/env python3
"""
Code Chunker - Chunks C/C++ source files and saves metadata to a JSON file.

This script takes a directory path as a command line argument, finds all .c and .h
files in that directory, chunks them using the CodeSplitter class, and saves the
results to a JSON file.

Usage:
    python code-chunker.py <path_to_source_directory>

Example:
    python code-chunker.py ../week05/xv6-riscv
"""

import os
import sys
import json
from pathlib import Path

# Import the CodeSplitter
try:
    # Using importlib to handle the hyphen in the filename
    import importlib.util
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
    print("pip install llama-index tree-sitter tree-sitter-languages")
    sys.exit(1)

def find_source_files(directory):
    """Find all .c and .h files in directory and subdirectories."""
    source_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.c', '.h')):
                source_files.append(os.path.join(root, file))
    return source_files

def process_file(file_path, base_dir):
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
            language="c",
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

def main():
    # Check if a directory path was provided
    if len(sys.argv) < 2:
        print("Error: No directory path provided.")
        print(f"Usage: python {sys.argv[0]} <path_to_source_directory>")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    
    # Check if the directory exists
    if not os.path.isdir(dir_path):
        print(f"Error: Directory '{dir_path}' does not exist.")
        sys.exit(1)
    
    # Get the absolute path and base directory name
    abs_dir_path = os.path.abspath(dir_path)
    base_dir_name = os.path.basename(abs_dir_path)
    
    # Find all source files
    source_files = find_source_files(abs_dir_path)
    print(f"Found {len(source_files)} source files in {abs_dir_path}")
    
    # Process each file and collect chunks
    all_chunks = []
    for file_path in source_files:
        print(f"Processing {file_path}")
        chunks = process_file(file_path, abs_dir_path)
        all_chunks.extend(chunks)
        print(f"  - Generated {len(chunks)} chunks")
    
    # Create the output file path
    output_file = f"{base_dir_name}-chunks.json"
    
    # Save the chunks to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\nSuccessfully processed {len(source_files)} files.")
    print(f"Generated {len(all_chunks)} chunks.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()