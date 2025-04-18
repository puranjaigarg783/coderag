# Enhanced CodeSplitter with File and Line Metadata

This implementation enhances the LlamaIndex `CodeSplitter` class to store file names and source code line numbers in the metadata of each chunk. This is particularly useful for source code RAG applications where traceability back to the original source files is important.

## Features

- **File Path Information**: Stores full path, relative path, and base filename in chunk metadata
- **Line Number Tracking**: Records the start and end line numbers for each code chunk
- **Backward Compatibility**: Maintains the same API as the original CodeSplitter
- **Flexible Configuration**: Accepts file path information either through constructor or document metadata
- **Proper Node Relationships**: Uses RelatedNodeInfo objects for node relationships to avoid Pydantic warnings

## Files

- `code-meta.py`: The enhanced CodeSplitter implementation
- `code-meta-example.py`: Example script demonstrating usage with a hardcoded C code example
- `code-meta-test.py`: Test script that takes a C source file path as a command line argument

## Usage

### Basic Usage

```python
from code_meta import CodeSplitter
from llama_index.core.schema import Document

# Create a document with file path information
document = Document(
    text=source_code,
    metadata={
        "filepath": "/path/to/source/file.c",
        "base_dir": "/path/to/source"
    }
)

# Initialize the CodeSplitter
code_splitter = CodeSplitter(language="c")

# Get nodes with enhanced metadata
nodes = code_splitter.get_nodes_from_documents([document])

# Access the enhanced metadata
for node in nodes:
    print(f"File: {node.metadata.get('filename')}")
    print(f"Path: {node.metadata.get('filepath')}")
    print(f"Relative path: {node.metadata.get('relpath')}")
    print(f"Lines: {node.metadata.get('start_line')} to {node.metadata.get('end_line')}")
```

### Using the Test Script

The `code-meta-test.py` script allows you to test the enhanced CodeSplitter with any C source file:

```bash
# Process a C file and analyze its chunks
python code-meta-test.py /path/to/source/file.c

# Example with a specific file
python code-meta-test.py ../../week05/sample-code/echo.c
```

The script will:
1. Process the specified C file
2. Split it into chunks with file and line metadata
3. Display information about each chunk
4. Demonstrate metadata filtering capabilities
5. Provide a summary of the results

This is particularly useful for testing the enhanced CodeSplitter with different C files to ensure it works correctly with various code structures and styles.

### Alternative Configuration

You can also provide file path information when initializing the CodeSplitter:

```python
# Initialize with file path information
code_splitter = CodeSplitter(
    language="c",
    filepath="/path/to/source/file.c",
    base_dir="/path/to/source"
)

# Create a simple document without path metadata
document = Document(text=source_code)

# The nodes will still have the file path metadata
nodes = code_splitter.get_nodes_from_documents([document])
```

### Filtering by Metadata

The enhanced metadata enables powerful filtering capabilities:

```python
# Filter nodes by filename
filename_filtered = [
    node for node in nodes 
    if node.metadata.get("filename") == "main.c"
]

# Filter nodes by line range (e.g., find nodes containing line 42)
line_filtered = [
    node for node in nodes 
    if node.metadata.get("start_line", 0) <= 42 and node.metadata.get("end_line", 0) >= 42
]

# Filter nodes by relative path
path_filtered = [
    node for node in nodes 
    if "src/core" in node.metadata.get("relpath", "")
]
```

## Metadata Fields

Each chunk includes the following metadata:

- `filename`: Base filename (e.g., "main.c")
- `filepath`: Full path to the file (e.g., "/home/user/project/src/main.c")
- `relpath`: Relative path from base directory (e.g., "src/main.c")
- `start_line`: First line number in the chunk (1-based)
- `end_line`: Last line number in the chunk (1-based)

## Requirements

- llama-index
- tree-sitter
- tree-sitter-languages

## Implementation Details

The enhanced CodeSplitter works by:

1. Tracking byte offsets for each line in the source code
2. Converting byte positions to line numbers during the chunking process
3. Storing file path and line information in the metadata of each chunk
4. Preserving this metadata when creating TextNode objects
5. Using proper RelatedNodeInfo objects for node relationships

This implementation maintains all the functionality of the original CodeSplitter while adding the enhanced metadata capabilities.

## Handling Pydantic Warnings

The original implementation might produce Pydantic serialization warnings like:

```
PydanticSerializationUnexpectedValue: Expected `RelatedNodeInfo` but got `str` with value '...' - serialized value may not be as expected
```

These warnings occur because LlamaIndex expects node relationships to use `RelatedNodeInfo` objects, but the original implementation uses string IDs. Our enhanced version fixes this by:

1. Importing the `RelatedNodeInfo` class from llama_index.core.schema
2. Creating proper `RelatedNodeInfo` objects for previous/next relationships
3. Setting these objects in the node relationships dictionary

This approach eliminates the Pydantic warnings while maintaining the same functionality.