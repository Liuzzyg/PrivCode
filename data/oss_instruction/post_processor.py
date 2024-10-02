import json
import pdb
from tqdm import tqdm  # Import tqdm for progress bar

def strip_leading_colons_and_newlines(text):
    """Removes leading colons and any newline characters before the actual content starts."""
    index = 0
    while index < len(text) and text[index] in [':', '\n', ' ', 'E', 'N', 'D']:
        index += 1
    return text[index:]

def find_next_block(text, label):
    """Helper function to find the next occurrence of a label and its content."""
    label_start = text.find(label)
    if label_start == -1:
        return None, -1  # No label found

    label_end = text.find("[END]", label_start)
    if label_end == -1:
        label_end = len(text)
    
    content = text[label_start + len(label):label_end].strip()
    return content, label_start

def is_invalid_content(text):
    """Checks if the content consists only of '\n' and '[END]', or contains other invalid tags."""
    # Remove all occurrences of '\n' and '[END]'
    cleaned_text = text.replace('\n', '').replace('[END]', '').strip()
    
    # If after cleaning the text is empty, it's invalid
    if cleaned_text == '':
        return True
    
    # Check for presence of cross-label content or unwanted tags like '[code]' or '[content]'
    if '[Problem Description]' in text or '[Solution]' in text or '[code]' in text or '[content]' in text:
        return True
    
    return False

def process_instruction_data(input_file, output_file):
    # First, count the total number of lines for progress bar initialization
    total_lines = sum(1 for _ in open(input_file, 'r'))

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Use tqdm to wrap the iteration for progress bar
        for line in tqdm(infile, total=total_lines, desc="Processing data"):
            try:
                data = json.loads(line.strip())
                index = data.get("index", None)
                completion = data.get("completion", "")

                # Extract Problem Description and Solution
                problem_text, problem_start_pos = find_next_block(completion, "[Problem Description]")
                solution_text, solution_start_pos = find_next_block(completion, "[Solution]")

                # Skip if either block is missing
                if problem_start_pos == -1 or solution_start_pos == -1:
                    continue

                # Clean the problem and solution texts (remove leading colons and newlines)
                problem_text = strip_leading_colons_and_newlines(problem_text)
                solution_text = strip_leading_colons_and_newlines(solution_text)

                # Check if problem_text or solution_text contain any invalid content
                if is_invalid_content(problem_text) or is_invalid_content(solution_text):
                    continue  # Skip this entry

                # If we have valid problem and solution, write them to the output
                output_data = {
                    "index": index,
                    "Problem": problem_text,
                    "Solution": solution_text
                }

                # Write to the output file
                outfile.write(json.dumps(output_data) + '\n')

            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")


input_file = 'data/oss_instruction/synthetic_numpy_instruction_data_25k.jsonl'
output_file = 'data/processed_instruction_data_25k.jsonl'
process_instruction_data(input_file, output_file)
