import ast
import json
import random
from datasets import load_dataset
from tqdm import tqdm
import jsonlines
import sys

# Set recursion limit
sys.setrecursionlimit(5000)  # Adjust as needed, 2000 is an example

# Get built-in classes
def get_builtin_classes():
    builtin_classes = {cls.__name__: cls for cls in (int, str, list, dict, set, tuple, bool, float)}
    return builtin_classes

# Extract API calls from code
def extract_apis(code, max_recursion_depth=1000):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}, {}, {}, {}
    
    # Structures for storing API-related information
    imported_modules = {}
    imported_names = {}
    variable_map = {}
    class_map = {}
    api_dict = {}
    non_api_dict = {}

    builtin_classes = get_builtin_classes()

    class ApiExtractor(ast.NodeVisitor):
        def __init__(self, max_depth):
            self.aliases = {'np': 'numpy'}
            self.current_depth = 0
            self.max_depth = max_depth

        def visit(self, node):
            # Skip if recursion depth exceeds the maximum allowed depth
            if self.current_depth >= self.max_depth:
                # Logging or printing to indicate the depth limit was reached, if needed
                print('maximum depth is reached, exit!!!!!!')
                return  # Exit early if maximum depth is reached

            self.current_depth += 1
            try:
                super().visit(node)
            finally:
                self.current_depth -= 1  # Ensure depth is decreased even if an error occurs

        def visit_Import(self, node):
            # Handle import statements
            for alias in node.names:
                imported_modules[alias.name] = alias.asname if alias.asname else alias.name
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            # Handle from ... import ... statements
            if node.module:
                for alias in node.names:
                    imported_names[alias.name] = (node.module, alias.asname if alias.asname else alias.name)
            self.generic_visit(node)

        def visit_Assign(self, node):
            # Handle assignment statements and track variable mapping
            if isinstance(node.value, ast.Call):
                func_name = self.get_func_name(node.value.func)
                if isinstance(node.targets[0], ast.Name):
                    variable_map[node.targets[0].id] = func_name
            self.generic_visit(node)

        def visit_Call(self, node):
            func_name = self.get_func_name(node.func)
            
            if isinstance(node.func, ast.Attribute):
                parts = func_name.split('.')
                base = parts[0].split('(', 1)[0]
                method_call = f"{'.'.join(parts[1:])}{self.get_call_args(node)}"
                
                if base in variable_map:
                    obj_init = variable_map[base]
                    api_call = f"{obj_init}.{method_call}"
                elif base in imported_modules:
                    api_call = f"{imported_modules[base]}.{method_call}"
                elif base in imported_names:
                    api_call = f"{imported_names[base]}.{method_call}"
                elif base in self.aliases:
                    api_call = f"{self.aliases[base]}.{method_call}"
                else:
                    api_call = func_name
                    
                self.add_api_call(api_call, func_name, node)
                
            elif isinstance(node.func, ast.Name):
                # Handle calls with known aliases
                if func_name in self.aliases:
                    api_call = f"{self.aliases[func_name]}.{self.get_call_args(node)}"
                else:
                    self.add_api_call(func_name, func_name, node)

            self.generic_visit(node)

        def get_func_name(self, node):
            """Helper function to get function name from AST nodes."""
            if isinstance(node, ast.Attribute):
                return f"{self.get_func_name(node.value)}.{node.attr}"
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return node.value
            else:
                return "unknown"

        def get_call_args(self, node):
            # Get call arguments
            args = ', '.join(ast.dump(arg) for arg in node.args)
            return f"({args})"

        def add_api_call(self, api_call, func_name, node):
            if api_call not in api_dict:
                api_dict[api_call] = []
            api_dict[api_call].append({
                "lineno": node.lineno,
                "col_offset": node.col_offset,
                "args": [ast.dump(arg) for arg in node.args]
            })

    # Traverse AST and extract API calls
    extractor = ApiExtractor(max_recursion_depth)
    extractor.visit(tree)

    return api_dict, non_api_dict, variable_map, class_map

# Randomly sample from the dataset
def sample_dataset(dataset, num_samples):
    sampled_indices = random.sample(range(len(dataset)), num_samples)
    return [dataset[i] for i in sampled_indices]

# Process samples and save
def filter_numpy_related_code(dataset, output_file, max_recursion_depth=5000):
    numpy_related_count = 0

    with jsonlines.open(output_file, mode='w') as writer:
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Samples"):

            code = sample['content']
            api_dict, _, _, _ = extract_apis(code, max_recursion_depth=max_recursion_depth)

            # Check if there are numpy-related API calls
            numpy_related = any(
                api_call.startswith("numpy.") or "numpy" in api_call
                for api_call in api_dict.keys()
            )

            # If there are numpy-related API calls, save the code snippet
            if numpy_related:
                writer.write({
                    "index": idx,
                    "content": code,
                    "api_calls": api_dict
                })
                numpy_related_count += 1

            # Print progress every 1000 samples
            if idx % 10000 == 0:
                print(f"Processed {idx} samples, found {numpy_related_count} numpy-related code snippets.")

    print(f"Extraction complete. Found {numpy_related_count} numpy-related code snippets.")

if __name__ == "__main__":
    # dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split='train')
    dataset = load_dataset("bigcode/starcoderdata", data_dir="python", split="train")
    # dataset = sample_dataset(dataset, 300000)
    filter_numpy_related_code(dataset, "/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy.jsonl", max_recursion_depth=1000)
    # filter_numpy_related_code(dataset, "starcoderdata_numpy.jsonl", max_recursion_depth=-1)
