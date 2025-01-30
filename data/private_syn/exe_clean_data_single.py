import json
import subprocess
import os
import tempfile
from tqdm import tqdm
from datasets import load_dataset
import argparse
import pdb
import shutil


# Function to create a dictionary from problems to languages for fast lookup
def create_problem_language_mapping(original_data):
    problem_language_map = {}
    for original_sample in original_data:
        problem = original_sample.get('problem', '').strip().lower()
        language = original_sample.get('lang', 'unknown')
        problem_language_map[problem] = language
    return problem_language_map

# Function to run code in various languages
def is_code_runnable(code_str, language, temp_dir):
    """
    Executes code for a specific language in a temporary directory.
    Supports: python, typescript, swift, cpp, java, rust, php, shell, csharp.
    """

    try:
        if not code_str.strip():
            return False
        first_line = code_str.splitlines()[0].strip().lower()
    except IndexError:
        return False
        
    valid_languages = {'python', 'typescript', 'swift', 'cpp', 'java', 'rust', 'php', 'shell', 'csharp'}
    if first_line in valid_languages:
        # Remove the invalid language header
        code_str = '\n'.join(code_str.splitlines()[1:]).strip()


    temp_dir = os.path.abspath(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    # pdb.set_trace()

    try:
        if language.lower() == "python":
            # Run Python code
            result = subprocess.run(
                ["python", "-c", code_str],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "typescript":
            # Save code to a temporary .ts file and run with tsc + node
            temp_file = os.path.join(temp_dir, "temp.ts")
            with open(temp_file, "w") as f:
                f.write(code_str)
            subprocess.run(["tsc", temp_file], timeout=5, check=True)
            result = subprocess.run(
                ["node", temp_file.replace(".ts", ".js")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "swift":
            # Save code to a temporary .swift file and execute
            temp_file = os.path.join(temp_dir, "temp.swift")
            with open(temp_file, "w") as f:
                f.write(code_str)
            result = subprocess.run(
                ["swift", temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "cpp":
            # Save code to a temporary .cpp file and compile + run
            temp_file = os.path.join(temp_dir, "temp.cpp")
            binary_file = os.path.join(temp_dir, "temp")
            with open(temp_file, "w") as f:
                f.write(code_str)
            subprocess.run(["g++", temp_file, "-o", binary_file], timeout=5, check=True)
            result = subprocess.run(
                [binary_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "java":
            # Save code to a temporary .java file and compile + run
            temp_file = os.path.join(temp_dir, "Temp.java")
            with open(temp_file, "w") as f:
                f.write(code_str)
            subprocess.run(["javac", temp_file], timeout=5, check=True)
            result = subprocess.run(
                ["java", "-cp", temp_dir, "Temp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "rust":
            # Save code to a temporary .rs file and compile + run
            temp_file = os.path.join(temp_dir, "temp.rs")
            binary_file = os.path.join(temp_dir, "temp")
            with open(temp_file, "w") as f:
                f.write(code_str)
            subprocess.run(["rustc", temp_file, "-o", binary_file], timeout=5, check=True)
            result = subprocess.run(
                [binary_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "php":
            # Save code to a temporary .php file and execute
            temp_file = os.path.join(temp_dir, "temp.php")
            with open(temp_file, "w") as f:
                f.write(code_str)
            result = subprocess.run(
                ["php", temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "shell":
            # Save code to a temporary .sh file and execute
            temp_file = os.path.join(temp_dir, "temp.sh")
            with open(temp_file, "w") as f:
                f.write(code_str)
            result = subprocess.run(
                ["bash", temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return result.returncode == 0

        elif language.lower() == "csharp":
            # Save code to a temporary .cs file, compile, and execute with dotnet
            temp_file = os.path.join(temp_dir, "Program.cs")
            project_dir = os.path.join(temp_dir, "CSharpProject")
            os.makedirs(project_dir, exist_ok=True)
            with open(os.path.join(project_dir, "Program.cs"), "w") as f:
                f.write(code_str)
            subprocess.run(["dotnet", "new", "console", "-o", project_dir, "--force"], timeout=10, check=True)
            subprocess.run(["dotnet", "run", "-p", project_dir], timeout=10, check=True)
            return True

        else:
            # Unsupported language
            return False

    except Exception:
        return False
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)  # 删除整个临时目录及其内容
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Function to clean and filter the data
def clean_data_with_executability(input_file, output_file):
    temp_dir = os.path.dirname(output_file)
    temp_dir = os.path.join(temp_dir, 'temp')
    cleaned_data = []
    language_classes = set()

    # Load dataset
    dataset = load_dataset(
        'ise-uiuc/Magicoder-OSS-Instruct-75K',
        split='train',
    )
    dataset = dataset.train_test_split(test_size=0.74, seed=42)
    test_data = dataset['test']

    print('Successfully loaded magicoder test data!')
    problem_language_map = create_problem_language_mapping(test_data)
    # pdb.set_trace()

    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, total=total_lines, desc="Processing data"):
            data = json.loads(line)
            problem = data.get('problem', '')
            solution = data.get('generated_solution', '')
            code_sections = solution.split('```')
            if len(code_sections) > 2:
                cleaned_solution = code_sections[1].strip()

                matched_language = problem_language_map.get(problem.strip().lower(), 'unknown')
                language_classes.add(matched_language)

                # Run code and validate executability
                # pdb.set_trace()
                if is_code_runnable(cleaned_solution, matched_language, temp_dir):
                    cleaned_data.append({
                        'language': matched_language,
                        'problem': problem,
                        'solution': "'''" + cleaned_solution + "'''",
                    })

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("Unique languages matched in the data:")
    print(language_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--output_file", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    args = parser.parse_args()

    clean_data_with_executability(args.input_file, args.output_file)

    print(f"Data cleaning and filtering completed. Cleaned data with executability saved to {args.output_file}.")
