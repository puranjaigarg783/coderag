# Code RAG Query

This tool allows querying language models (LLMs) with code chunks to answer questions about the code.

## Requirements

- Python 3.6+
- LlamaIndex packages

## Installation

1. Install the base LlamaIndex package:

```
pip install llama-index
```

2. Install the model-specific packages based on which models you want to use:

For OpenAI (gpt-3.5):

```
pip install llama-index-llms-openai
```

For Anthropic (claude-3.7):

```
pip install llama-index-llms-anthropic
```

## API Keys

You'll need to set the appropriate API key as an environment variable:

- For OpenAI: `export OPENAI_API_KEY="your-api-key-here"`
- For Anthropic: `export ANTHROPIC_API_KEY="your-api-key-here"`

## Usage

```
python code-rag-query.py query <model> <chunks_file> "<question>"
```

Where:

- `<model>` is either `gpt-3.5` or `claude-3.7`
- `<chunks_file>` is a JSON file containing code chunks
- `<question>` is the question you want to ask about the code

Example:

```
python code-rag-query.py query gpt-3.5 xv6-riscv-chunks.json "What does the xshort function do?"
```

## JSON Chunks Format

The JSON chunks file should be in the following format:

```json
[
  {
    "filepath": "/full/path/to/file.c",
    "filename": "file.c",
    "relpath": "relative/path/to/file.c",
    "start_line": 1,
    "end_line": 66,
    "length": 66,
    "content": "/* Source code content */\n..."
  },
  {
    "filepath": "/full/path/to/another_file.c",
    "filename": "another_file.c",
    "relpath": "relative/path/another_file.c",
    "start_line": 10,
    "end_line": 20,
    "length": 11,
    "content": "/* More source code */\n..."
  }
]
```
