import gradio as gr
import subprocess
import sys
import os
import threading
import time
import uuid
import glob
import shutil
from pathlib import Path

default_command = "bigcodebench.evaluate"
# is_running = False

def generate_command(
    jsonl_file, split, subset, parallel,
    min_time_limit, max_as_limit, max_data_limit, max_stack_limit,
    check_gt_only, no_gt
):
    command = [default_command]
    
    if jsonl_file is not None:
        # Copy the uploaded file to the current directory
        local_filename = os.path.basename(jsonl_file.name)
        shutil.copy(jsonl_file.name, local_filename)
        command.extend(["--samples", local_filename])
    
    command.extend(["--split", split, "--subset", subset])
    
    if parallel is not None and parallel != 0:
        command.extend(["--parallel", str(int(parallel))])
    
    command.extend([
        "--min-time-limit", str(min_time_limit),
        "--max-as-limit", str(int(max_as_limit)),
        "--max-data-limit", str(int(max_data_limit)),
        "--max-stack-limit", str(int(max_stack_limit))
    ])
    
    if check_gt_only:
        command.append("--check-gt-only")
    
    if no_gt:
        command.append("--no-gt")
    
    return " ".join(command)


def cleanup_previous_files(jsonl_file):
    if jsonl_file is not None:
        file_list = ['Dockerfile', 'app.py', 'README.md', os.path.basename(jsonl_file.name), "__pycache__"]
    else:
        file_list = ['Dockerfile', 'app.py', 'README.md', "__pycache__"]
    for file in glob.glob("*"):
        try:
            if file not in file_list and not file.startswith("bigcode"):
                os.remove(file)
        except Exception as e:
            print(f"Error during cleanup of {file}: {e}")

def find_result_file():
    json_files = glob.glob("*.json")
    if json_files:
        return max(json_files, key=os.path.getmtime)
    return None

def run_bigcodebench(command):
    # global is_running
    # if is_running:
    #     yield "A command is already running. Please wait for it to finish.\n"
    #     return
    # is_running = True

    # try:
    yield f"Executing command: {command}\n"
    
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    def kill_process():
        if process.poll() is None:  # If the process is still running
            process.terminate()
            # is_running = False
            yield "Process terminated after 12 minutes timeout.\n"

    # Start a timer to kill the process after 12 minutes
    timer = threading.Timer(720, kill_process)
    timer.start()
    
    for line in process.stdout:
        yield line
    
    # process.wait()
    
    timer.cancel()
    
    if process.returncode != 0:
        yield f"Error: Command exited with status {process.returncode}\n"
    
    yield "Evaluation completed.\n"
    
    result_file = find_result_file()
    if result_file:
        yield f"Result file found: {result_file}\n"
    else:
        yield "No result file found.\n"
    # finally:
    #     is_running = False

def stream_logs(command, jsonl_file=None):
    # global is_running
        
    # if is_running:
    #     yield "A command is already running. Please wait for it to finish.\n"
    #     return
    
    cleanup_previous_files(jsonl_file)
    yield "Cleaned up previous files.\n"

    log_content = []
    for log_line in run_bigcodebench(command):
        log_content.append(log_line)
        yield "".join(log_content)
        
# with gr.Blocks() as demo:
#     gr.Markdown("# BigCodeBench Evaluator")
    
#     with gr.Row():
#         jsonl_file = gr.File(label="Upload JSONL file", file_types=[".jsonl"])
#         split = gr.Dropdown(choices=["complete", "instruct"], label="Split", value="complete")
#         subset = gr.Dropdown(choices=["hard", "full"], label="Subset", value="hard")
    
#     with gr.Row():
#         parallel = gr.Number(label="Parallel (optional)", precision=0)
#         min_time_limit = gr.Number(label="Min Time Limit", value=1, precision=1)
#         max_as_limit = gr.Number(label="Max AS Limit", value=25*1024, precision=0)
    
#     with gr.Row():
#         max_data_limit = gr.Number(label="Max Data Limit", value=25*1024, precision=0)
#         max_stack_limit = gr.Number(label="Max Stack Limit", value=10, precision=0)
#         check_gt_only = gr.Checkbox(label="Check GT Only")
#         no_gt = gr.Checkbox(label="No GT")
    
#     command_output = gr.Textbox(label="Command", value=default_command, interactive=False)
#     with gr.Row():
#         submit_btn = gr.Button("Run Evaluation")
#         download_btn = gr.DownloadButton(label="Download Result")
#     log_output = gr.Textbox(label="Execution Logs", lines=20)
    
#     input_components = [
#         jsonl_file, split, subset, parallel,
#         min_time_limit, max_as_limit, max_data_limit, max_stack_limit,
#         check_gt_only, no_gt
#     ]
    
#     for component in input_components:
#         component.change(generate_command, inputs=input_components, outputs=command_output)
        
    
#     def start_evaluation(command, jsonl_file, subset, split):
#         extra = subset + "_" if subset != "full" else ""
#         if jsonl_file is not None:
#             result_path = os.path.basename(jsonl_file.name).replace(".jsonl", f"_{extra}eval_results.json")
#         else:
#             result_path = None

#         for log in stream_logs(command, jsonl_file):
#             if jsonl_file is not None:
#                 yield log, gr.update(value=result_path, label=result_path), gr.update()
#             else:
#                 yield log, gr.update(), gr.update()
#         result_file = find_result_file()
#         if result_file:
#             return gr.update(label="Evaluation completed. Result file found."), gr.update(value=result_file)
#                     # gr.Button(visible=False)#,
#                     # gr.DownloadButton(label="Download Result", value=result_file, visible=True))
#         else:
#             return gr.update(label="Evaluation completed. No result file found."), gr.update(value=result_path)
#                     # gr.Button("Run Evaluation", visible=True),
#                     # gr.DownloadButton(visible=False))
#     submit_btn.click(start_evaluation,
#                  inputs=[command_output, jsonl_file, subset, split],
#                  outputs=[log_output, download_btn])

# REPO_ID = "bigcode/bigcodebench-evaluator"
# HF_TOKEN = os.environ.get("HF_TOKEN", None)
# API = HfApi(token=HF_TOKEN)

# def restart_space():
#     API.restart_space(repo_id=REPO_ID, token=HF_TOKEN)

# demo.queue(max_size=300).launch(share=True, server_name="0.0.0.0", server_port=7860)
# scheduler = BackgroundScheduler()
# scheduler.add_job(restart_space, "interval", hours=3) # restarted every 3h as backup in case automatic updates are not working
# scheduler.start()