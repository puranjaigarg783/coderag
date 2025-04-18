import json
import sys

def load_chunks(filename):
    """Load chunks from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def get_file_line_mapping(chunks):
    """
    Create a mapping from relpath to a set of line numbers contained in the chunks.
    This allows us to track which lines from which files are included in each JSON.
    """
    file_lines = {}
    for chunk in chunks:
        filepath = chunk["relpath"]
        if filepath not in file_lines:
            file_lines[filepath] = set()
        
        # Add all lines in this chunk to the set
        for line_num in range(chunk["start_line"], chunk["end_line"]):
            file_lines[filepath].add(line_num)
    
    return file_lines

def analyze_overlap(file_lines1, file_lines2):
    """
    Analyze the overlap between two sets of files with their line numbers.
    Handles cases where a chunk in file1 may span multiple chunks in file2.
    """
    overlap_count = 0
    only_in_file1_count = 0
    only_in_file2_count = 0
    
    # Get all unique filepaths from both files
    all_filepaths = set(file_lines1.keys()).union(set(file_lines2.keys()))
    
    # Process each filepath
    for filepath in all_filepaths:
        lines1 = file_lines1.get(filepath, set())
        lines2 = file_lines2.get(filepath, set())
        
        # Calculate overlap and exclusive lines for this file
        file_overlap = lines1.intersection(lines2)
        file_only_in_1 = lines1 - lines2
        file_only_in_2 = lines2 - lines1
        
        overlap_count += len(file_overlap)
        only_in_file1_count += len(file_only_in_1)
        only_in_file2_count += len(file_only_in_2)
    


    d = {
        "overlap_count": overlap_count,
        "only_in_file1_count": only_in_file1_count,
        "only_in_file2_count": only_in_file2_count,
        "lines_in_file1": overlap_count + only_in_file1_count,
        "lines_in_file2": overlap_count + only_in_file2_count
    }

    print(d)
    return d


def print_summary(chunks):
    """Print a summary of a chunk set."""
    # Count unique files
    filepaths = set()
    total_lines = 0
    
    for chunk in chunks:
        filepaths.add(chunk["filepath"])
        # Use length field for total lines count
        total_lines += chunk["end_line"] - chunk["start_line"]
    
    print(f"Total number of files: {len(filepaths)}")
    print(f"Total number of chunks: {len(chunks)}")
    print(f"Total number of lines: {total_lines}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python retrieval-perf.py <ground-truth.json> <test-set.json>")
        return
    
    ground_truth_file = sys.argv[1]
    test_set_file = sys.argv[2]
    
    # Load the chunk data from files
    ground_truth_chunks = load_chunks(ground_truth_file)
    test_set_chunks = load_chunks(test_set_file)
    
    # Print summary for each set
    print(f"Ground Truth Summary:")
    print_summary(ground_truth_chunks)
    print()
    
    print(f"Test Set Summary:")
    print_summary(test_set_chunks)
    print()
    
    # Create mappings of which files and lines are in each JSON
    ground_truth_file_lines = get_file_line_mapping(ground_truth_chunks)
    test_set_file_lines = get_file_line_mapping(test_set_chunks)
    
    # Analyze the overlap between the two sets
    results = analyze_overlap(ground_truth_file_lines, test_set_file_lines)
    
    # Calculate metrics
    relevant_retrieved = results["overlap_count"]
    ground_truth_lines = results["lines_in_file1"]
    retrieved_lines = results["lines_in_file2"]
    
    # Calculate recall: (relevant lines retrieved) / (ground truth lines)
    if ground_truth_lines == 0:
        recall = 0.0
    else:
        recall = relevant_retrieved / ground_truth_lines
    
    # Calculate precision: (relevant lines retrieved) / (total lines retrieved)
    if retrieved_lines == 0:
        precision = 0.0
    else:
        precision = relevant_retrieved / retrieved_lines
    
    # Calculate F1 score: harmonic mean of precision and recall
    if precision == 0.0 and recall == 0.0:
        f1_score = 0.0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    
    # Print metrics
    print(f"Retrieval Performance Metrics:")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1: {f1_score:.4f}")

if __name__ == "__main__":
    main()