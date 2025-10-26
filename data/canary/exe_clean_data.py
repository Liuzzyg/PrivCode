import json
import os
import subprocess
import shutil
from tqdm import tqdm
import argparse

def is_code_runnable(code_str, temp_dir):
    """
    Checks if Python code is runnable.
    """
    try:
        if not code_str.strip():
            return False

        # Remove any potential language headers
        first_line = code_str.splitlines()[0].strip().lower()
        if first_line in {'python'}:
            code_str = '\n'.join(code_str.splitlines()[1:]).strip()

        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        temp_file = os.path.join(temp_dir, "temp_code.py")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code_str)

        try:
            # Run the Python code
            result = subprocess.run(
                ["python", temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    finally:
        try:
            # Clean up the temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Error during cleanup: {e}")

def clean_data(input_file, output_file):
    """
    Sequentially processes the data to filter executable Python code.
    """
    temp_dir = os.path.join(os.path.dirname(output_file), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    cleaned_data = []
    last_processed = None

    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with tqdm(total=len(lines), desc="Processing data") as pbar:
        for line in lines:
            try:
                data = json.loads(line)
                problem = data.get("problem", "")
                solution = data.get("generated_solution", "")
                code_sections = solution.split("```")

                # Check if code block exists and if it contains '```bash'
                if len(code_sections) > 2 and '```bash' not in solution:
                    cleaned_solution = code_sections[1].strip()
                    matched_language = "python"

                    # Check if the code is executable
                    if is_code_runnable(cleaned_solution, temp_dir):
                        entry = {
                            "language": matched_language,
                            "problem": problem,
                            "solution": f"'''{cleaned_solution}''',"
                        }
                        cleaned_data.append(entry)
                        last_processed = entry  # Record the last successfully processed entry
            except Exception:
                if last_processed:
                    print(f"Interrupted. Last processed entry: {json.dumps(last_processed, ensure_ascii=False)}")
                raise
            finally:
                pbar.update(1)

    # Save the cleaned data
    with open(output_file, "w", encoding="utf-8") as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Data cleaning completed. Cleaned data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input.json")
    parser.add_argument("--output_file", type=str, default="output.json")
    args = parser.parse_args()

    clean_data(args.input_file, args.output_file)
