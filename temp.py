import ast
import subprocess

def check_python_syntax(code_snippet):
    try:
        tree = ast.parse(code_snippet)
        return True
    except SyntaxError as e:
        print(f"Python syntax error: {e}")
        return False

def check_java_syntax(code_snippet):
    try:
        # 将代码保存到临时文件
        with open("temp.java", "w") as f:
            f.write(code_snippet)
        # 使用 javac 编译 Java 代码
        subprocess.run(["javac", "temp.java"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Java syntax error: {e}")
        return False

def check_cpp_syntax(code_snippet):
    try:
        # 将代码保存到临时文件
        with open("temp.cpp", "w") as f:
            f.write(code_snippet)
        # 使用 g++ 编译 C++ 代码
        subprocess.run(["g++", "-fsyntax-only", "temp.cpp"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"C++ syntax error: {e}")
        return False

def check_syntax(language, code_snippet):
    if language == "python":
        return check_python_syntax(code_snippet)
    elif language == "java":
        return check_java_syntax(code_snippet)
    elif language == "cpp":
        return check_cpp_syntax(code_snippet)
    else:
        print("Unsupported language")
        return False

# 示例用法
# code_snippets = """
# def my_function():
#     print("Hello, world!")
# """

# if check_syntax("python", code_snippets):
#     print("Python code is valid.")

java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}
"""

if check_syntax("java", java_code):
    print("Java code is valid.")

# cpp_code = """
# #include <iostream>
# int main() {
#     std::cout << "Hello, world!" << std::endl;
#     return 0;
# }
# """

# if check_syntax("cpp", cpp_code):
#     print("C++ code is valid.")
