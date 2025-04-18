#!/usr/bin/env python3
"""
Using LlamaIndex with Tree-Sitter for C code parsing.
This program demonstrates how to:
1. Set up tree-sitter and the C language parser
2. Configure llama_index CodeSplitter
3. Process C code using llama_index
"""

import os
import sys
from pathlib import Path

# Required packages:
# pip install llama-index tree-sitter

try:
    from llama_index.core.schema import Document
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import CodeSplitter
    from tree_sitter import Language, Parser
except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required packages:")
    print("pip install llama-index tree-sitter")
    sys.exit(1)

def setup_tree_sitter():
    """
    Set up tree-sitter for C language parser.
    """
    try:
        # Directory to store tree-sitter languages
        languages_dir = Path("./tree-sitter-langs")
        languages_dir.mkdir(exist_ok=True)
        
        # Path to the built language library
        library_path = languages_dir / "languages.so"
        
        # Check if we need to build the language
        if not library_path.exists():
            print("Setting up tree-sitter C parser...")
            
            # Clone the C grammar repository if it doesn't exist
            c_repo_path = languages_dir / "tree-sitter-c"
            if not c_repo_path.exists():
                os.system(f"git clone https://github.com/tree-sitter/tree-sitter-c {c_repo_path}")
            
            # Build the language library
            Language.build_library(
                str(library_path),
                [str(c_repo_path)]
            )
            print("Tree-sitter C parser setup complete!")
        
        # Set environment variable for tree-sitter to find the languages
        # This is a key step - LlamaIndex's CodeSplitter now looks for this environment variable
        os.environ["LLAMA_INDEX_TREE_SITTER_LIB_PATH"] = str(library_path)
        
        # Load the C language to verify it works
        C_LANGUAGE = Language(str(library_path), 'c')
        print("Successfully loaded C language parser")
        return C_LANGUAGE, str(library_path)
    except Exception as e:
        print(f"Error setting up tree-sitter: {e}")
        import traceback
        traceback.print_exc()
        return None, ""

def verify_parser(c_language):
    """Verify that the parser works by parsing a simple C program."""
    parser = Parser()
    parser.set_language(c_language)
    
    test_code = b"int main() { return 0; }"
    tree = parser.parse(test_code)
    
    print("Parser test:")
    print(f"- Root node type: {tree.root_node.type}")
    print(f"- First child type: {tree.root_node.children[0].type if tree.root_node.children else 'None'}")
    
    return tree.root_node.type == "translation_unit"

def process_c_code_with_llama_index(c_code):
    """Process C code using LlamaIndex with tree-sitter."""
    try:
        # Create a Document from the C code
        document = Document(text=c_code)
        
        # Configure the CodeSplitter with tree-sitter C language
        # Note: lib_path is now handled via environment variable LLAMA_INDEX_TREE_SITTER_LIB_PATH
        code_splitter = CodeSplitter(
            language="c",
            chunk_lines=10,  # Adjust as needed
            chunk_lines_overlap=5,  # Adjust as needed
            max_chars=256,  # Adjust as needed
        )
        
        # Split the document into nodes
        nodes = code_splitter.get_nodes_from_documents([document])
        
        print(f"\nDocument split into {len(nodes)} nodes")
        
        # Print info about each node
        for i, node in enumerate(nodes):
            print(f"\nNode {i+1}:")
            print(f"- Text length: {len(node.text)}")
            # Print first 50 chars of the node text
            #preview = node.text[:50].replace('\n', '\\n')
            print(f"- Meta: {node.metadata.get("function_name")}")
            preview = node.text
            print(f"- Preview: {preview}...")
            
            # Print metadata
            print("- Metadata:")
            for key, value in node.metadata.items():
                print(f"  • {key}: {value}")
        
        # Create a simple index from the nodes
        # (In a real application, you would configure a proper embedding model)
        index = VectorStoreIndex(nodes)
        
        return nodes, index
        
    except Exception as e:
        print(f"Error processing with LlamaIndex: {e}")
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
    
    try:
        # Setup tree-sitter C parser
        #c_language, library_path = setup_tree_sitter()
        #if c_language is None:
        #    print("Failed to set up tree-sitter. Exiting.")
        #    return

        #C_LANGUAGE = Language(str(library_path), 'c')
        
        # Verify parser works
        #if not verify_parser(C_LANGUAGE):
        #    print("Parser verification failed. Exiting.")
        #    return
            
        # Process the C code with LlamaIndex
        print("\nProcessing C code with LlamaIndex...")
        nodes, index = process_c_code_with_llama_index(c_code)
        
        if nodes and index:
            print("\nSuccessfully processed C code with LlamaIndex!")
            
            # Demonstrate a simple query (without LLM for this example)
            print("\nExample metadata-based retrieval:")
            filtered_nodes = [
                node for node in nodes 
                if node.metadata.get("function_name") == "factorial"
            ]
            
            if filtered_nodes:
                print(f"Found {len(filtered_nodes)} nodes related to 'factorial':")
                for node in filtered_nodes:
                    print(f"- Line range: {node.metadata.get('start_line', 'N/A')} to {node.metadata.get('end_line', 'N/A')}")
            else:
                print("No nodes found for 'factorial' function.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
