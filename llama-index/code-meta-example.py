#!/usr/bin/env python3
"""
Example demonstrating the enhanced CodeSplitter with file path and line number metadata.
This program shows how to:
1. Use the enhanced CodeSplitter from code-meta.py
2. Process code with file path information
3. Access and utilize the enhanced metadata
"""

import os
import sys
from pathlib import Path

# Import the enhanced CodeSplitter
try:
    # Import the enhanced CodeSplitter from the local file
    # Using importlib to handle the hyphen in the filename
    import importlib.util
    spec = importlib.util.spec_from_file_location("code_meta", "code-meta.py")
    code_meta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_meta)
    CodeSplitter = code_meta.CodeSplitter
    
    from llama_index.core.schema import Document
    from llama_index.core import VectorStoreIndex
    from tree_sitter import Language, Parser
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure you have the required packages installed:")
    print("pip install llama-index tree-sitter tree-sitter-languages")
    sys.exit(1)

def process_c_code_with_enhanced_splitter(c_code, filepath, base_dir):
    """Process C code using the enhanced CodeSplitter with file path metadata."""
    try:
        # Create a Document from the C code with file path information
        document = Document(
            text=c_code,
            metadata={
                "filepath": filepath,
                "base_dir": base_dir
            }
        )
        
        # Configure the enhanced CodeSplitter
        code_splitter = CodeSplitter(
            language="c",
            chunk_lines=10,  # Adjust as needed
            chunk_lines_overlap=5,  # Adjust as needed
            max_chars=256,  # Adjust as needed
        )
        
        # Split the document into nodes with enhanced metadata
        nodes = code_splitter.get_nodes_from_documents([document])
        
        print(f"\nDocument split into {len(nodes)} nodes")
        
        # Print info about each node with enhanced metadata
        for i, node in enumerate(nodes):
            print(f"\nNode {i+1}:")
            print(f"- Text length: {len(node.text)}")
            
            # Print file path information
            print("- File information:")
            print(f"  • Filename: {node.metadata.get('filename', 'N/A')}")
            print(f"  • Filepath: {node.metadata.get('filepath', 'N/A')}")
            print(f"  • Relative path: {node.metadata.get('relpath', 'N/A')}")
            
            # Print line number information
            print("- Line information:")
            print(f"  • Start line: {node.metadata.get('start_line', 'N/A')}")
            print(f"  • End line: {node.metadata.get('end_line', 'N/A')}")
            
            # Print text preview
            preview = node.text[:50].replace('\n', '\\n')
            print(f"- Preview: {preview}...")
            
        # Create a simple index from the nodes
        index = VectorStoreIndex(nodes)
        
        # Demonstrate filtering by file and line information
        print("\nDemonstrating metadata filtering capabilities:")
        
        # Filter nodes by filename
        filename_filtered = [node for node in nodes if node.metadata.get("filename") == os.path.basename(filepath)]
        print(f"- Found {len(filename_filtered)} nodes from file '{os.path.basename(filepath)}'")
        
        # Filter nodes by line range (e.g., find nodes containing line 20)
        line_filtered = [
            node for node in nodes 
            if node.metadata.get("start_line", 0) <= 20 and node.metadata.get("end_line", 0) >= 20
        ]
        print(f"- Found {len(line_filtered)} nodes containing line 20")
        
        return nodes, index
        
    except Exception as e:
        print(f"Error processing with enhanced CodeSplitter: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    # Example C code
    c_code = """
    #include <stdio.h>
    
    /**
     * A function to calculate the factorial of a number
     * @param n The number to calculate factorial for
     * @return The factorial of n
     */
    int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    
    /**
     * A function to check if a number is prime
     * @param n The number to check
     * @return 1 if prime, 0 otherwise
     */
    int is_prime(int n) {
        if (n <= 1) return 0;
        if (n <= 3) return 1;
        
        if (n % 2 == 0 || n % 3 == 0) return 0;
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0)
                return 0;
        }
        
        return 1;
    }
    
    /**
     * Main function
     */
    int main() {
        int num = 5;
        
        printf("Factorial of %d is %d\\n", num, factorial(num));
        
        if (is_prime(num)) {
            printf("%d is a prime number\\n", num);
        } else {
            printf("%d is not a prime number\\n", num);
        }
        
        return 0;
    }
    """
    
    # Example file path information
    filepath = "/path/to/source/example.c"
    base_dir = "/path/to/source"
    
    try:
        # Process the C code with enhanced CodeSplitter
        print("\nProcessing C code with enhanced CodeSplitter...")
        nodes, index = process_c_code_with_enhanced_splitter(c_code, filepath, base_dir)
        
        if nodes and index:
            print("\nSuccessfully processed C code with enhanced metadata!")
            
            # Demonstrate a simple query based on metadata
            print("\nExample metadata-based retrieval:")
            
            # Find nodes from the factorial function
            factorial_nodes = [
                node for node in nodes 
                if "factorial" in node.text and node.metadata.get("start_line", 0) < 15
            ]
            
            if factorial_nodes:
                print(f"Found {len(factorial_nodes)} nodes related to 'factorial':")
                for node in factorial_nodes:
                    print(f"- Line range: {node.metadata.get('start_line', 'N/A')} to {node.metadata.get('end_line', 'N/A')}")
                    print(f"- File: {node.metadata.get('filename', 'N/A')}")
            else:
                print("No nodes found for 'factorial' function.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()