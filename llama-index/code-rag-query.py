#!/usr/bin/env python3
import json
import sys
import argparse
import os
from typing import List, Dict, Any, Optional, Tuple

def check_requirements(model: str) -> bool:
    """
    Check if the required packages for the specified model are installed.
    Print installation instructions if they are not.
    
    Args:
        model: The model to check requirements for
        
    Returns:
        True if all requirements are met, False otherwise
    """
    try:
        import llama_index
    except ImportError:
        print("Error: LlamaIndex is not installed.")
        print("Please install it with: pip install llama-index")
        return False

    if model == "gpt-3.5":
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError:
            print("Error: OpenAI module for LlamaIndex is not installed.")
            print("Please install it with: pip install llama-index-llms-openai")
            return False
    
    if model == "claude-3.7":
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError:
            print("Error: Anthropic module for LlamaIndex is not installed.")
            print("Please install it with: pip install llama-index-llms-anthropic")
            return False
    
    return True

def judge_answer(model: str, ground_truth_file: str, test_answer_file: str) -> None:
    """
    Judge how well a test answer matches a ground truth answer using an LLM.
    
    Args:
        model: The LLM to use for judging (gpt-3.5 or claude-3.7)
        ground_truth_file: Path to the file containing the ground truth answer
        test_answer_file: Path to the file containing the test answer
    """
    # Check for required packages before proceeding
    if not check_requirements(model):
        sys.exit(1)
        
    # Load the ground truth and test answers
    ground_truth = load_text_file(ground_truth_file)
    test_answer = load_text_file(test_answer_file)
    
    if ground_truth is None or test_answer is None:
        sys.exit(1)
    
    # Check for empty or very short test answers first
    if not test_answer or len(test_answer.strip()) < 10:
        print(f"Model: {model}")
        print("Score: 0")
        print("Explanation: No answer or answer too short")
        return
    
    # Create the prompt for the LLM to judge the answer
    prompt = create_judge_prompt(ground_truth, test_answer)
    
    # Query the appropriate model and get the response
    response = None
    if model == "gpt-3.5":
        response = query_gpt(prompt)
    elif model == "claude-3.7":
        response = query_claude(prompt)
    
    if response:
        # Parse the response to extract the score and explanation
        score, explanation = parse_judge_response(response)
        
        # Print the results
        print(f"Model: {model}")
        print(f"Score: {score}")
        print(f"Explanation: {explanation}")
    else:
        # Fall back to the simple algorithm if the LLM query failed
        score, explanation = calculate_score(ground_truth, test_answer)
        print(f"Model: {model} (using fallback scoring)")
        print(f"Score: {score}")
        print(f"Explanation: {explanation}")

def create_judge_prompt(ground_truth: str, test_answer: str) -> str:
    """
    Create a prompt for the LLM to judge how well a test answer matches a ground truth.
    
    Args:
        ground_truth: The ground truth answer text
        test_answer: The test answer text to evaluate
    
    Returns:
        A prompt string for the LLM
    """
    prompt = """I need you to judge how well a test answer matches a ground truth answer.
    
Please analyze the similarity and assign a score from 0-3 based on these criteria:
- 3: Very close to the ground truth
- 2: Partially close to ground truth, but missing details or contains non-relevant information
- 1: Not close to the ground truth
- 0: No answer or answer too short

Ground Truth Answer:
```
{ground_truth}
```

Test Answer:
```
{test_answer}
```

Provide your assessment in this exact format:
SCORE: [number 0-3]
EXPLANATION: [your reasoning]
""".format(ground_truth=ground_truth, test_answer=test_answer)
    
    print("--- Judge Prompt ---")
    print(prompt)
    print("--------------------")
    
    return prompt

def parse_judge_response(response: str) -> Tuple[int, str]:
    """
    Parse the LLM's response to extract the score and explanation.
    
    Args:
        response: The LLM's response to the judge prompt
    
    Returns:
        A tuple containing the score (0-3) and an explanation
    """
    try:
        # Look for the score pattern "SCORE: [number]"
        score_match = None
        for line in response.split('\n'):
            if line.strip().startswith("SCORE:"):
                score_match = line.strip()
                break
        
        if score_match:
            score_str = score_match.split("SCORE:")[1].strip()
            score = int(score_str)
            
            # Validate the score is within range
            if score < 0:
                score = 0
            elif score > 3:
                score = 3
        else:
            # If no score found, default to 1
            score = 1
        
        # Look for the explanation pattern "EXPLANATION: [text]"
        explanation_match = None
        for line in response.split('\n'):
            if line.strip().startswith("EXPLANATION:"):
                explanation_match = line.strip()
                break
        
        if explanation_match:
            explanation = explanation_match.split("EXPLANATION:")[1].strip()
        else:
            # Try to extract any text after "SCORE:" as the explanation
            parts = response.split("SCORE:")
            if len(parts) > 1:
                possible_explanation = parts[1].strip()
                # Remove the score number
                explanation = ' '.join(possible_explanation.split()[1:])
            else:
                explanation = "Unable to parse explanation from LLM response"
        
        return score, explanation
    
    except Exception as e:
        print(f"Error parsing judge response: {str(e)}")
        # Fall back to a neutral score and explanation
        return 1, "Error parsing LLM response"

def calculate_score(ground_truth: str, test_answer: str) -> Tuple[int, str]:
    """
    Calculate a score based on how well the test answer matches the ground truth.
    
    Args:
        ground_truth: The ground truth answer text
        test_answer: The test answer text
        
    Returns:
        A tuple containing the score (0-3) and an explanation
    """
    # Check for empty or very short answers first
    if not test_answer or len(test_answer.strip()) < 10:
        return 0, "No answer or answer too short"
    
    # Simple criteria for scoring:
    # 3 - Very close to the ground truth
    # 2 - Partially close to ground truth, but missing details or contains non-relevant information
    # 1 - Not close
    # 0 - No answer
    
    # Convert texts to lowercase for better comparison
    ground_truth_lower = ground_truth.lower()
    test_answer_lower = test_answer.lower()
    
    # Calculate simple content overlap
    gt_words = set(ground_truth_lower.split())
    ta_words = set(test_answer_lower.split())
    
    # Get intersection and unique words
    common_words = gt_words.intersection(ta_words)
    gt_unique = gt_words - ta_words
    ta_unique = ta_words - gt_words
    
    # Calculate overlap percentages
    if len(gt_words) > 0:
        gt_coverage = len(common_words) / len(gt_words)
    else:
        gt_coverage = 0
        
    if len(ta_words) > 0:
        precision = len(common_words) / len(ta_words)
    else:
        precision = 0
    
    # Make scoring decisions based on coverage and precision
    if gt_coverage > 0.8 and precision > 0.8:
        score = 3
        explanation = "Very close to the ground truth"
    elif gt_coverage > 0.5 and precision > 0.5:
        score = 2
        explanation = "Partially close to ground truth, but missing some details or contains non-relevant information"
    else:
        score = 1
        explanation = "Not close to the ground truth"
    
    return score, explanation

def load_text_file(file_path: str) -> Optional[str]:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        The text content, or None if there was an error
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Code RAG Query - Use code chunks with LLMs to answer user questions")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query an LLM with code chunks and a question")
    query_parser.add_argument("model", choices=["gpt-3.5", "claude-3.7"], help="LLM to use for the query")
    query_parser.add_argument("chunks_file", help="JSON file containing code chunks")
    query_parser.add_argument("question", help="Question to ask the LLM")
    
    # Judge command
    judge_parser = subparsers.add_parser("judge", help="Judge the similarity between a test answer and a ground truth answer")
    judge_parser.add_argument("model", choices=["gpt-3.5", "claude-3.7"], help="LLM to use for judging the answer")
    judge_parser.add_argument("ground_truth_file", help="File containing the ground truth answer")
    judge_parser.add_argument("test_answer_file", help="File containing the test answer to evaluate")
    
    args = parser.parse_args()
    
    if args.command == "query":
        # Check for required packages before proceeding
        if check_requirements(args.model):
            query_llm(args.model, args.chunks_file, args.question)
    elif args.command == "judge":
        # Check for required packages before proceeding
        if check_requirements(args.model):
            judge_answer(args.model, args.ground_truth_file, args.test_answer_file)
    else:
        parser.print_help()

def query_llm(model: str, chunks_file: str, question: str) -> None:
    """
    Load code chunks, create a prompt, query the specified language model, and print the response.
    
    Args:
        model: The language model to use (gpt-3.5 or claude-3.7)
        chunks_file: Path to a JSON file containing code chunks
        question: The question to ask the language model about the code
    """
    # 1. Load the code chunks from the JSON file
    chunks = load_code_chunks(chunks_file)
    if not chunks:
        sys.exit(1)
    
    # 2. Create the prompt with code chunks and question
    prompt = create_prompt(chunks, question)
    
    # 3. Query the appropriate model and get the response
    response = None
    if model == "gpt-3.5":
        response = query_gpt(prompt)
    elif model == "claude-3.7":
        response = query_claude(prompt)
    
    if response:
        # 4. Print the response
        print(response)
    else:
        sys.exit(1)

def load_code_chunks(chunks_file: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load code chunks from a JSON file.
    
    Args:
        chunks_file: Path to the JSON file containing code chunks
    
    Returns:
        A list of code chunk dictionaries, or None if there was an error
    """
    try:
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        return chunks
    except FileNotFoundError:
        print(f"Error: Chunks file '{chunks_file}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Chunks file '{chunks_file}' is not valid JSON")
        return None
    except Exception as e:
        print(f"Error loading chunks file: {str(e)}")
        return None

def create_prompt(chunks: List[Dict[str, Any]], question: str) -> str:
    """
    Create a prompt that includes the code chunks and the user's question.
    
    Args:
        chunks: A list of code chunk dictionaries
        question: The question to ask about the code
    
    Returns:
        A prompt string that includes the code chunks and question
    """
    prompt = "I will provide you with code chunks and a question. Please analyze the code and answer the question based on the provided code.\n\n"
    prompt += "Code Chunks:\n\n"
    
    for i, chunk in enumerate(chunks):
        prompt += f"Chunk {i+1} (File: {chunk['filename']}, Lines {chunk['start_line']}-{chunk['end_line']}):\n"
        prompt += f"```\n{chunk['content']}\n```\n\n"
    
    prompt += f"Question: {question}\n\n"
    prompt += "Please provide a detailed answer to the question based only on the code chunks provided."
    
    print("--- Prompt ---")
    print(prompt)
    print("--------------")

    return prompt

def query_gpt(prompt: str) -> Optional[str]:
    """
    Query the OpenAI GPT-3.5 API using LlamaIndex.
    
    Args:
        prompt: The prompt to send to the API
    
    Returns:
        The API response as a string, or None if there was an error
    """
    # Import here to avoid errors if the package isn't installed
    from llama_index.llms.openai import OpenAI
    
    # This requires the OpenAI API key to be set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        return None
    
    try:
        # Initialize the OpenAI LLM through LlamaIndex
        llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
        
        # Complete the prompt
        response = llm.complete(prompt)
        
        return str(response)
    except Exception as e:
        print(f"Error querying GPT-3.5: {str(e)}")
        return None

def query_claude(prompt: str) -> Optional[str]:
    """
    Query the Anthropic Claude-3.7 API using LlamaIndex.
    
    Args:
        prompt: The prompt to send to the API
    
    Returns:
        The API response as a string, or None if there was an error
    """
    # Import here to avoid errors if the package isn't installed
    from llama_index.llms.anthropic import Anthropic
    
    # This requires the Anthropic API key to be set in the environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        return None
    
    try:
        # Initialize the Claude LLM through LlamaIndex
        llm = Anthropic(model="claude-3-sonnet-20240229", api_key=api_key)
        
        # Complete the prompt
        response = llm.complete(prompt)
        
        return str(response)
    except Exception as e:
        print(f"Error querying Claude-3.7: {str(e)}")
        return None

if __name__ == "__main__":
    main()