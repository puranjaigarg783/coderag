import json
import sys

def load_chunks(filename):
    """Load chunks from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def get_file_line_mapping(chunks):
    """
    Create a mapping from filepath to a set of line numbers contained in the chunks.
    This allows us to track which lines from which files are included in each JSON.
    """
    file_lines = {}
    for chunk in chunks:
        filepath = chunk["filepath"]
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
    
    return {
        "overlap_count": overlap_count,
        "only_in_file1_count": only_in_file1_count,
        "only_in_file2_count": only_in_file2_count,
        "lines_in_file1": overlap_count + only_in_file1_count,
        "lines_in_file2": overlap_count + only_in_file2_count
    }

def calculate_percentages(results):
    """Calculate percentage overlaps."""
    total_lines_file1 = results["lines_in_file1"]
    total_lines_file2 = results["lines_in_file2"]
    overlap_count = results["overlap_count"]
    only_in_file1_count = results["only_in_file1_count"]
    only_in_file2_count = results["only_in_file2_count"]
    
    if total_lines_file1 > 0:
        pct_overlap_in_file1 = (overlap_count / total_lines_file1) * 100
        pct_only_in_file1 = (only_in_file1_count / total_lines_file1) * 100
    else:
        pct_overlap_in_file1 = 0
        pct_only_in_file1 = 0
    
    if total_lines_file2 > 0:
        pct_overlap_in_file2 = (overlap_count / total_lines_file2) * 100
        pct_only_in_file2 = (only_in_file2_count / total_lines_file2) * 100
    else:
        pct_overlap_in_file2 = 0
        pct_only_in_file2 = 0
    
    return {
        "total_lines_file1": total_lines_file1,
        "total_lines_file2": total_lines_file2,
        "overlap_count": overlap_count,
        "only_in_file1_count": only_in_file1_count,
        "only_in_file2_count": only_in_file2_count,
        "pct_overlap_in_file1": pct_overlap_in_file1,
        "pct_only_in_file1": pct_only_in_file1,
        "pct_overlap_in_file2": pct_overlap_in_file2,
        "pct_only_in_file2": pct_only_in_file2
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare-chunks.py <file1.json> <file2.json>")
        return
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    # Load the chunk data from files
    chunks1 = load_chunks(file1)
    chunks2 = load_chunks(file2)
    
    # Create mappings of which files and lines are in each JSON
    file_lines1 = get_file_line_mapping(chunks1)
    file_lines2 = get_file_line_mapping(chunks2)
    
    # Analyze the overlap between the two sets
    results = analyze_overlap(file_lines1, file_lines2)
    
    # Calculate percentages
    stats = calculate_percentages(results)
    
    # Print the results
    print(f"Comparison of {file1} and {file2}:")
    print()
    print(f"File 1 has {stats['total_lines_file1']} total lines in chunks")
    print(f"File 2 has {stats['total_lines_file2']} total lines in chunks")
    print()
    print(f"Intersection (lines in both files): {stats['overlap_count']} lines")
    print(f"  {stats['pct_overlap_in_file1']:.2f}% of File 1")
    print(f"  {stats['pct_overlap_in_file2']:.2f}% of File 2")
    print()
    print(f"Only in File 1: {stats['only_in_file1_count']} lines ({stats['pct_only_in_file1']:.2f}%)")
    print(f"Only in File 2: {stats['only_in_file2_count']} lines ({stats['pct_only_in_file2']:.2f}%)")

if __name__ == "__main__":
    main()