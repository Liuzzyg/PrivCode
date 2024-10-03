import os
import logging
import time
import datetime
import gradio as gr
from threading import Thread, Lock
import datasets
from huggingface_hub import snapshot_download, WebhooksServer, WebhookPayload, RepoCard
from gradio_leaderboard import Leaderboard, ColumnFilter, SelectColumns
from apscheduler.schedulers.background import BackgroundScheduler

lock = Lock()

# Start ephemeral Spaces on PRs (see config in README.md)
from gradio_space_ci.webhook import IS_EPHEMERAL_SPACE, SPACE_ID, configure_space_ci

from src.display.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    # INTRODUCTION_TEXT,
    TITLE,
    ABOUT_TEXT,
    SUBMISSION_TEXT_3,
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    AutoEvalColumn,
    fields,
    EvalQueueColumn
)
from src.envs import (
    API,
    EVAL_REQUESTS_PATH,
    RESULT_REPO,
    DATA_VERSION,
    DATA_REPO,
    HARD_RESULT_REPO,
    ELO_REPO,
    HARD_ELO_REPO,
    SOLVE_REPO,
    HARD_SOLVE_REPO,
    HF_TOKEN,
    QUEUE_REPO,
    REPO_ID,
    VOTES_REPO,
    VOTES_PATH,
    HF_HOME,
)
from src.populate import get_evaluation_queue_df, get_leaderboard_df
from src.execute import generate_command, default_command, stream_logs, find_result_file
from src.tools.plots import plot_elo_mle, plot_solve_rate
# from src.voting.vote_system import VoteManager, run_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Start ephemeral Spaces on PRs (see config in README.md)
from gradio_space_ci.webhook import IS_EPHEMERAL_SPACE, SPACE_ID, configure_space_ci

# Convert the environment variable "LEADERBOARD_FULL_INIT" to a boolean value, defaulting to True if the variable is not set.
# This controls whether a full initialization should be performed.
DO_FULL_INIT = True # os.getenv("LEADERBOARD_FULL_INIT", "True") == "True"
NEW_DATA_ON_LEADERBOARD = True
LEADERBOARD_DF = None
HARD_LEADERBOARD_DF = None
ELO_TASK_DF = None
ELO_BENCH_DF = None
HARD_ELO_TASK_DF = None
HARD_ELO_BENCH_DF = None
COMPLETE_SOLVE_DF = None
INSTRUCT_SOLVE_DF = None
HARD_COMPLETE_SOLVE_DF = None
HARD_INSTRUCT_SOLVE_DF = None

DATA = datasets.load_dataset(DATA_REPO, "default", cache_dir=HF_HOME, split=DATA_VERSION,
                             verification_mode="no_checks")


def filter_data(data, keyword):
    if not keyword:
        return data
    filtered_data = [item for item in data if keyword.lower() in item['complete_prompt'].lower()]
    return filtered_data


def update_display(search_keyword, index, show_test):
    filtered_data = filter_data(DATA, search_keyword)
    
    if not filtered_data:
        return ["No data available. Check the search criteria."] + [""] * 4 + [0, gr.update(maximum=0, value=0)]
    
    max_index = len(filtered_data) - 1
    index = min(max(0, index), max_index)
    
    task_id = filtered_data[index]['task_id']
    snippet1 = filtered_data[index]['complete_prompt']
    snippet2 = filtered_data[index]['instruct_prompt']
    # snippet3 = filtered_data[index]['canonical_solution'] if show_solution else ""
    snippet4 = filtered_data[index]['test'] if show_test else ""
    
    return [
        task_id,
        snippet1,
        snippet2,
        # snippet3,
        snippet4,
        len(filtered_data),
        gr.update(maximum=max_index, value=index)
    ]

def restart_space():
    API.restart_space(repo_id=REPO_ID, token=HF_TOKEN)


def time_diff_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        diff = end_time - start_time
        logging.info(f"Time taken for {func.__name__}: {diff} seconds")
        return result

    return wrapper


@time_diff_wrapper
def download_dataset(repo_id, local_dir, repo_type="dataset", max_attempts=3, backoff_factor=1.5):
    """Download dataset with exponential backoff retries."""
    attempt = 0
    while attempt < max_attempts:
        try:
            logging.info(f"Downloading {repo_id} to {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type=repo_type,
                tqdm_class=None,
                etag_timeout=30,
                max_workers=8,
            )
            logging.info("Download successful")
            return
        except Exception as e:
            wait_time = backoff_factor**attempt
            logging.error(f"Error downloading {repo_id}: {e}, retrying in {wait_time}s")
            time.sleep(wait_time)
            attempt += 1
    raise Exception(f"Failed to download {repo_id} after {max_attempts} attempts")

def get_latest_data_leaderboard(
    leaderboard_initial_df = None,
    hard_leaderboard_initial_df = None,
    elo_task_df = None,
    elo_bench_df = None,
    hard_elo_task_df = None,
    hard_elo_bench_df = None,
    complete_solve_df = None,
    instruct_solve_df = None,
    hard_complete_solve_df = None,
    hard_instruct_solve_df = None
    ):
    global NEW_DATA_ON_LEADERBOARD
    global LEADERBOARD_DF
    global HARD_LEADERBOARD_DF
    global ELO_TASK_DF
    global ELO_BENCH_DF
    global HARD_ELO_TASK_DF
    global HARD_ELO_BENCH_DF
    global COMPLETE_SOLVE_DF
    global INSTRUCT_SOLVE_DF
    global HARD_COMPLETE_SOLVE_DF
    global HARD_INSTRUCT_SOLVE_DF

    if NEW_DATA_ON_LEADERBOARD:
        print("Leaderboard updated at reload!")
        leaderboard_dataset = datasets.load_dataset(
            RESULT_REPO, 
            "default", 
            split="train", 
            cache_dir=HF_HOME, 
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset 
            verification_mode="no_checks"
        )
        LEADERBOARD_DF = get_leaderboard_df(
            leaderboard_dataset=leaderboard_dataset, 
            cols=COLS,
        )
        hard_leaderboard_dataset = datasets.load_dataset(
            HARD_RESULT_REPO, 
            "default", 
            split="train", 
            cache_dir=HF_HOME, 
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset 
            verification_mode="no_checks"
        )
        hard_leaderboard_df = get_leaderboard_df(
            leaderboard_dataset=hard_leaderboard_dataset, 
            cols=COLS,
        )
        HARD_LEADERBOARD_DF = hard_leaderboard_df
        
        elo_task_df = datasets.load_dataset(
            ELO_REPO,
            "default", 
            split="task_no_tie", 
            cache_dir=HF_HOME, 
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset 
            verification_mode="no_checks"
        ).to_pandas()
        elo_bench_df = datasets.load_dataset(
            ELO_REPO,
            "default", 
            split="benchmark_tie", 
            cache_dir=HF_HOME, 
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset 
            verification_mode="no_checks"
        ).to_pandas()
        ELO_TASK_DF = elo_task_df
        ELO_BENCH_DF = elo_bench_df
        
        hard_elo_task_df = datasets.load_dataset(
            HARD_ELO_REPO,
            "default", 
            split="task_no_tie", 
            cache_dir=HF_HOME, 
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset 
            verification_mode="no_checks"
        ).to_pandas()
        hard_elo_bench_df = datasets.load_dataset(
            HARD_ELO_REPO,
            "default", 
            split="benchmark_tie", 
            cache_dir=HF_HOME, 
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset 
            verification_mode="no_checks"
        ).to_pandas()
        HARD_ELO_TASK_DF = hard_elo_task_df
        HARD_ELO_BENCH_DF = hard_elo_bench_df
        
        complete_solve_df = datasets.load_dataset(
            SOLVE_REPO,
            "default",
            split="complete",
            cache_dir=HF_HOME,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset
            verification_mode="no_checks"
        ).to_pandas()
        instruct_solve_df = datasets.load_dataset(
            SOLVE_REPO,
            "default",
            split="instruct",
            cache_dir=HF_HOME,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset
            verification_mode="no_checks"
        ).to_pandas()
        COMPLETE_SOLVE_DF = complete_solve_df
        INSTRUCT_SOLVE_DF = instruct_solve_df
        
        hard_complete_solve_df = datasets.load_dataset(
            HARD_SOLVE_REPO,
            "default",
            split="complete",
            cache_dir=HF_HOME,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset
            verification_mode="no_checks"
        ).to_pandas()
        hard_instruct_solve_df = datasets.load_dataset(
            HARD_SOLVE_REPO,
            "default",
            split="instruct",
            cache_dir=HF_HOME,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, # Uses the cached dataset
            verification_mode="no_checks"
        ).to_pandas()        
        HARD_COMPLETE_SOLVE_DF = hard_complete_solve_df
        HARD_INSTRUCT_SOLVE_DF = hard_instruct_solve_df
        
        NEW_DATA_ON_LEADERBOARD = False

    else:
        LEADERBOARD_DF = leaderboard_initial_df
        # HARD_LEADERBOARD_DF = hard_leaderboard_initial_df
        ELO_TASK_DF = elo_task_df
        # ELO_BENCH_DF = elo_bench_df
        # HARD_ELO_TASK_DF = hard_elo_task_df
        HARD_ELO_BENCH_DF = hard_elo_bench_df
        COMPLETE_SOLVE_DF = complete_solve_df
        # INSTRUCT_SOLVE_DF = instruct_solve_df
        # HARD_COMPLETE_SOLVE_DF = hard_complete_solve_df
        HARD_INSTRUCT_SOLVE_DF = hard_instruct_solve_df
        
    return (LEADERBOARD_DF, HARD_LEADERBOARD_DF, ELO_TASK_DF, ELO_BENCH_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, COMPLETE_SOLVE_DF, INSTRUCT_SOLVE_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF)
    # return (HARD_LEADERBOARD_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF)


def init_space():
    """Initializes the application space, loading only necessary data."""

    # Always redownload the leaderboard DataFrame
    global LEADERBOARD_DF
    global HARD_LEADERBOARD_DF
    global ELO_TASK_DF
    global ELO_BENCH_DF
    global HARD_ELO_TASK_DF
    global HARD_ELO_BENCH_DF
    global COMPLETE_SOLVE_DF
    global INSTRUCT_SOLVE_DF
    global HARD_COMPLETE_SOLVE_DF
    global HARD_INSTRUCT_SOLVE_DF
    
    LEADERBOARD_DF, HARD_LEADERBOARD_DF, ELO_TASK_DF, ELO_BENCH_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, COMPLETE_SOLVE_DF, INSTRUCT_SOLVE_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF = get_latest_data_leaderboard()
    # HARD_LEADERBOARD_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF = get_latest_data_leaderboard()

    return (LEADERBOARD_DF, HARD_LEADERBOARD_DF, ELO_TASK_DF, ELO_BENCH_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, COMPLETE_SOLVE_DF, INSTRUCT_SOLVE_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF)
    # return (HARD_LEADERBOARD_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF)

# Initialize VoteManager
# vote_manager = VoteManager(VOTES_PATH, EVAL_REQUESTS_PATH, VOTES_REPO)


# Schedule the upload_votes method to run every 15 minutes
# schedule.every(15).minutes.do(vote_manager.upload_votes)

# Start the scheduler in a separate thread
# scheduler_thread = Thread(target=run_scheduler, args=(vote_manager,), daemon=True)
# scheduler_thread.start()

# Calls the init_space function with the `full_init` parameter determined by the `do_full_init` variable.
# This initializes various DataFrames used throughout the application, with the level of initialization detail controlled by the `do_full_init` flag.
LEADERBOARD_DF, HARD_LEADERBOARD_DF, ELO_TASK_DF, \
ELO_BENCH_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, \
COMPLETE_SOLVE_DF, INSTRUCT_SOLVE_DF, HARD_COMPLETE_SOLVE_DF, \
HARD_INSTRUCT_SOLVE_DF = init_space()
# HARD_LEADERBOARD_DF, HARD_ELO_TASK_DF, HARD_ELO_BENCH_DF, HARD_COMPLETE_SOLVE_DF, HARD_INSTRUCT_SOLVE_DF = init_space()

# Data processing for plots now only on demand in the respective Gradio tab
# def load_and_create_plots():
#     plot_df = create_plot_df(create_scores_df(LEADERBOARD_DF))
#     return plot_df

# Function to check if a user is logged in
def check_login(profile: gr.OAuthProfile | None) -> bool:
    if profile is None:
        return False
    return True

def init_leaderboard(dataframe):
    if dataframe is None or dataframe.empty:
        raise ValueError("Leaderboard DataFrame is empty or None.")
    return Leaderboard(
        value=dataframe,
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[c.name for c in fields(AutoEvalColumn) if c.displayed_by_default],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden or c.dummy],
            label="Select Columns to Display:",
        ),
        search_columns=[AutoEvalColumn.model.name],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        filter_columns=[
            ColumnFilter(AutoEvalColumn.type.name, type="checkboxgroup", label="Model Types"),
            ColumnFilter(AutoEvalColumn.openness.name, type="checkboxgroup", label="Openness"),
            ColumnFilter(AutoEvalColumn.size_range.name, type="dropdown", label="Model Size"),
            ColumnFilter(AutoEvalColumn.moe.name, type="checkboxgroup", label="Model Architecture"),
        ],
        bool_checkboxgroup_label="Hide models",
        interactive=False,
        )


def init_others(dataframe):
    if dataframe is None or dataframe.empty:
        raise ValueError("Gradio DataFrame is empty or None.")
    return gr.Dataframe(dataframe, visible=False)

main_block = gr.Blocks(css=custom_css)
with main_block as demo:
    with gr.Row(elem_id="header-row"):
        gr.HTML(TITLE + "<p>Total models: " + str(len(HARD_LEADERBOARD_DF))+ "</p>")
    
    # gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")    
    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.Tab("💎 Hard Set") as hard_tabs:
            with gr.TabItem("🏅 Benchmark", elem_id="llm-benchmark-tab-table", id="hard_bench"):
                hard_leaderboard = init_leaderboard(HARD_LEADERBOARD_DF)
                gr.Markdown(
                    """
                **Notes:**
                - For the limited compute, we now update the Hard Set leaderboard. (**We are open to sponsorship for more compute!**)
                - _Hard Set_ vs _Full Set_:
                    - <u>Hard Set</u>: A subset of ~150 BigCodeBench tasks which is more user-facing and challenging.
                    - <u>Full Set</u>: The full set of 1140 BigCodeBench tasks.
                - _Complete_ vs _Instruct_:
                    - <u>Complete</u>: Code Completion based on the (verbose) structured docstring. This split tests if the models are good at coding.
                    - <u>Instruct</u> (🔥Vibe Check🔥): Code Generation based on the (less verbose) NL-oriented instructions. This split tests if the models are really capable enough to understand human intents to code.
                - `Complete` and `Instruct` represent the calibrated Pass@1 score on the BigCodeBench benchmark splits.
                - `Average` is the average of `Complete` and `Instruct` when both are available.
                - `Elo Rating` represents the task-level Bootstrap of Maximum Likelihood Elo rating on the Complete + Instruct splits. The rating starts from 1000 and is bootstrapped 500 times. We only consider the models having both `Complete` and `Instruct` scores.
                - `#Act Params (B)` is the number of activated model parameters during inference.
                - Model providers have the responsibility to avoid data contamination. Models trained on close data can be affected by contamination.
                - For more details check the 📝 About section.
                """,
                    elem_classes="markdown-text",
                )
            
            with gr.TabItem("📊 Elo Rating", id="hard_elo"):
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("## (Task-level, No Tie, BigCodeBench-Complete) -- _Recommended_")
                        hard_task_elo_map = gr.Plot()
                        hard_elo_task_gr = init_others(HARD_ELO_TASK_DF)
                        demo.load(plot_elo_mle, [hard_elo_task_gr],
                                    hard_task_elo_map)
                    with gr.Group():
                        gr.Markdown("## (Benchmark-level, BigCodeBench-Complete)")
                        hard_bench_elo_map = gr.Plot()
                        hard_elo_bench_gr = init_others(HARD_ELO_BENCH_DF)
                        demo.load(plot_elo_mle, [hard_elo_bench_gr],
                                    hard_bench_elo_map)
                        
            with gr.TabItem("🧩 Solve Rate", id="hard_solve"):
                with gr.Column():
                    hard_complete_map = gr.Plot()
                    hard_complete_solve_gr = init_others(HARD_COMPLETE_SOLVE_DF)
                    demo.load(plot_solve_rate, [hard_complete_solve_gr,
                                                gr.Textbox("Complete", visible=False),
                                                gr.Number(10, visible=False),
                                                gr.Number(16, visible=False),
                                                ], hard_complete_map)
                    hard_instruct_map = gr.Plot()
                    hard_instruct_solve_gr = init_others(HARD_INSTRUCT_SOLVE_DF)
                    demo.load(plot_solve_rate, [hard_instruct_solve_gr,
                                                gr.Textbox("Instruct", visible=False),
                                                gr.Number(10, visible=False),
                                                gr.Number(16, visible=False),
                                                ], hard_instruct_map)
        with gr.Tab("🎯 Full Set") as full_tabs:
            with gr.TabItem("🏅 Benchmark", elem_id="llm-benchmark-tab-table", id="full_bench"):
                leaderboard = init_leaderboard(LEADERBOARD_DF)
                gr.Markdown(
                    """
                **Notes:**
                - _Complete_ vs _Instruct_:
                    - <u>Complete</u>: Code Completion based on the (verbose) structured docstring. This variant tests if the models are good at coding.
                    - <u>Instruct</u> (🔥Vibe Check🔥): Code Generation based on the (less verbose) NL-oriented instructions. This variant tests if the models are really capable enough to understand human intents to code.
                - `complete` and `instruct` represent the calibrated Pass@1 score on the BigCodeBench benchmark variants.
                - `elo_mle` represents the task-level Bootstrap of Maximum Likelihood Elo rating on the BigCodeBench-Complete split. The rating starts from 1000 and is bootstrapped 500 times.
                - `size` is the amount of activated model weight during inference.
                - Model providers have the responsibility to avoid data contamination. Models trained on close data can be affected by contamination.
                - For more details check the 📝 About section.
                """,
                    elem_classes="markdown-text",
                )
            
            with gr.TabItem("📊 Elo Rating", id="full_elo"):
                with gr.Column():
                    with gr.Group():
                        
                        gr.Markdown("## (Task-level, No Tie, BigCodeBench-Complete) -- _Recommended_")
                        task_elo_map = gr.Plot()
                        elo_task_gr = init_others(ELO_TASK_DF)
                        demo.load(plot_elo_mle, [elo_task_gr], task_elo_map)
                    with gr.Group():
                        gr.Markdown("## (Benchmark-level, BigCodeBench-Complete)")
                        bench_elo_map = gr.Plot()
                        elo_bench_gr = init_others(ELO_BENCH_DF)
                        demo.load(plot_elo_mle, [elo_bench_gr], bench_elo_map)
                    
            with gr.TabItem("🧩 Solve Rate", id="full_solve"):
                with gr.Column():
                    complete_map = gr.Plot()
                    complete_solve_gr = init_others(COMPLETE_SOLVE_DF)
                    demo.load(plot_solve_rate, [complete_solve_gr,
                                                gr.Textbox("Complete", visible=False),
                                                ], complete_map)
                    instruct_map = gr.Plot()
                    instruct_solve_gr = init_others(INSTRUCT_SOLVE_DF)
                    demo.load(plot_solve_rate, [instruct_solve_gr,
                                                gr.Textbox("Instruct", visible=False),
                                                ], instruct_map)
        with gr.TabItem("📝 About", id=3):
            gr.Markdown(ABOUT_TEXT, elem_classes="markdown-text")

        with gr.TabItem("🔎 Data Viewer", id="viewer"):
            search_input = gr.Textbox(label="Search by keyword")
            count_output = gr.Number(label="Number of filtered items")
            index_slider = gr.Slider(minimum=0, maximum=len(DATA)-1, step=1, label="Select Index")
            # show_solution = gr.Checkbox(label="Show Solution")
            show_test = gr.Checkbox(label="Show Test Cases")
            update_button = gr.Button("Update")
            next_button = gr.Button("Next")
            prev_button = gr.Button("Prev")
            
            task_id_output = gr.Textbox(label="Task ID")
            code_completion = gr.Code(language="python", label="Code Completion")
            nl_instruction = gr.Code(language="markdown", label="Natural Language Instruction")
            # solution = gr.Code(language="python", label="Solution")
            test_cases = gr.Code(language="python", label="Test Cases")
            
            update_button.click(
                update_display, 
                inputs=[search_input, index_slider, show_test],
                outputs=[task_id_output, code_completion, nl_instruction, test_cases, count_output, index_slider]
            )
            next_button.click(
                lambda search, index, show_test: update_display(search, index + 1, show_test),
                inputs=[search_input, index_slider, show_test],
                outputs=[task_id_output, code_completion, nl_instruction, test_cases, count_output, index_slider]
            )
            prev_button.click(
                lambda search, index, show_test: update_display(search, index - 1, show_test),
                inputs=[search_input, index_slider, show_test],
                outputs=[task_id_output, code_completion, nl_instruction, test_cases, count_output, index_slider]
            )
            # Initial load
            demo.load(
                update_display, 
                inputs=[search_input, index_slider, show_test],
                outputs=[task_id_output, code_completion, nl_instruction, test_cases, count_output, index_slider]
            )
            
        with gr.TabItem("🛠️ Code Execution (Beta)", id=5):
            gr.Markdown("""\
## Upload your [sanitized JSONL file](https://github.com/bigcode-project/bigcodebench?tab=readme-ov-file#code-post-processing) to evaluate

### Hard Set Ground Truth Pass Rate: 100%
### Full Set Ground Truth Pass Rate: 99.6%

### Note: The execution could be stuck, and we are working on it. Meanwhile, please clone this space to your own hub and run the evaluation there. If you notice that your execution is stuck, please restart the cloned space.
""")
            with gr.Row():
                jsonl_file = gr.File(label="Upload JSONL file", file_types=[".jsonl"])
                split = gr.Dropdown(choices=["complete", "instruct"], label="Split", value="complete")
                subset = gr.Dropdown(choices=["hard", "full"], label="Subset", value="hard")
            
            with gr.Row():
                parallel = gr.Number(label="Parallel (optional)", precision=0)
                min_time_limit = gr.Number(label="Min Time Limit", value=1, precision=1)
                max_as_limit = gr.Number(label="Max AS Limit", value=25*1024, precision=0)
            
            with gr.Row():
                max_data_limit = gr.Number(label="Max Data Limit", value=25*1024, precision=0)
                max_stack_limit = gr.Number(label="Max Stack Limit", value=10, precision=0)
                check_gt_only = gr.Checkbox(label="Check GT Only", value=False, visible=False)
                no_gt = gr.Checkbox(label="No GT", value=False, visible=False)
            
            command_output = gr.Textbox(label="Command", value=default_command, interactive=False)
            with gr.Row():
                submit_btn = gr.Button("Run Evaluation")
                download_btn = gr.DownloadButton(label="Download Result", visible=False)
            log_output = gr.Textbox(label="Execution Logs", lines=20)
            
            input_components = [
                jsonl_file, split, subset, parallel,
                min_time_limit, max_as_limit, max_data_limit, max_stack_limit,
                check_gt_only, no_gt
            ]
            
            for component in input_components:
                component.change(generate_command, inputs=input_components, outputs=command_output)
                
            
            def start_evaluation(command, jsonl_file, subset, split):
                lock.acquire()
                try:
                    extra = subset + "_" if subset != "full" else ""
                    if jsonl_file is not None:
                        result_path = os.path.basename(jsonl_file.name).replace(".jsonl", f"_{extra}eval_results.json")
                    else:
                        result_path = None

                    for log in stream_logs(command, jsonl_file):
                        if jsonl_file is not None and jsonl_file.name.endswith(".jsonl"):
                            yield log, gr.update(value=result_path, label=result_path, visible=True), gr.update(visible=False)
                        else:
                            yield log, gr.update(), gr.update()
                    result_file = find_result_file()
                    if result_file:
                        return gr.update(label="Evaluation completed. Result file found."), gr.update(value=result_file)
                                # gr.Button(visible=False)#,
                                # gr.DownloadButton(label="Download Result", value=result_file, visible=True))
                    else:
                        return gr.update(label="Evaluation completed. No result file found."), gr.update(value=result_path)
                                # gr.Button("Run Evaluation", visible=True),
                                # gr.DownloadButton(visible=False))
                finally:
                    lock.release()
            submit_btn.click(start_evaluation,
                        inputs=[command_output, jsonl_file, subset, split],
                        outputs=[log_output, download_btn, submit_btn])
        
        with gr.TabItem("🚀 Request", id=4):
            gr.Markdown(SUBMISSION_TEXT_3)
    
    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                lines=20,
                elem_id="citation-button",
                show_copy_button=True,
            )
                    
    main_block.load(fn=get_latest_data_leaderboard, inputs=[leaderboard, hard_leaderboard, elo_task_gr, elo_bench_gr, hard_elo_task_gr, hard_elo_bench_gr, complete_solve_gr, instruct_solve_gr, hard_complete_solve_gr, hard_instruct_solve_gr], outputs=[leaderboard, hard_leaderboard, elo_task_gr, elo_bench_gr, hard_elo_task_gr, hard_elo_bench_gr, complete_solve_gr, instruct_solve_gr, hard_complete_solve_gr, hard_instruct_solve_gr])
    # main_block.load(fn=get_latest_data_leaderboard, inputs=[hard_leaderboard, hard_elo_task_gr, hard_elo_bench_gr, hard_complete_solve_gr, hard_instruct_solve_gr], outputs=[hard_leaderboard, hard_elo_task_gr, hard_elo_bench_gr, hard_complete_solve_gr, hard_instruct_solve_gr])
    # leaderboard.change(fn=get_latest_data_queue, inputs=None, outputs=[finished_eval_table, running_eval_table, pending_eval_table])
    # pending_eval_table.change(fn=vote_manager.create_request_vote_df, inputs=[pending_eval_table], outputs=[pending_eval_table_votes])

main_block.queue(default_concurrency_limit=100)


def enable_space_ci_and_return_server(ui: gr.Blocks) -> WebhooksServer:
    # Taken from https://huggingface.co/spaces/Wauplin/gradio-space-ci/blob/075119aee75ab5e7150bf0814eec91c83482e790/src/gradio_space_ci/webhook.py#L61
    # Compared to original, this one do not monkeypatch Gradio which allows us to define more webhooks.
    # ht to Lucain!
    if SPACE_ID is None:
        print("Not in a Space: Space CI disabled.")
        return WebhooksServer(ui=main_block)

    if IS_EPHEMERAL_SPACE:
        print("In an ephemeral Space: Space CI disabled.")
        return WebhooksServer(ui=main_block)

    card = RepoCard.load(repo_id_or_path=SPACE_ID, repo_type="space")
    config = card.data.get("space_ci", {})
    print(f"Enabling Space CI with config from README: {config}")

    return configure_space_ci(
        blocks=ui,
        trusted_authors=config.get("trusted_authors"),
        private=config.get("private", "auto"),
        variables=config.get("variables", "auto"),
        secrets=config.get("secrets"),
        hardware=config.get("hardware"),
        storage=config.get("storage"),
    )

# Create webhooks server (with CI url if in Space and not ephemeral)
webhooks_server = enable_space_ci_and_return_server(ui=main_block)

# Add webhooks
@webhooks_server.add_webhook
def update_leaderboard(payload: WebhookPayload) -> None:
    """Redownloads the leaderboard dataset each time it updates"""
    if payload.repo.type == "dataset" and payload.event.action == "update":
        global NEW_DATA_ON_LEADERBOARD
        if NEW_DATA_ON_LEADERBOARD:
            return
        NEW_DATA_ON_LEADERBOARD = True

        for repo in [RESULT_REPO, HARD_RESULT_REPO, ELO_REPO, HARD_ELO_REPO, SOLVE_REPO, HARD_SOLVE_REPO]:
            datasets.load_dataset(
                repo, 
                "default", 
                cache_dir=HF_HOME, 
                download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD, 
                verification_mode="no_checks"
            )
        
        

webhooks_server.launch()

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", hours=3) # restarted every 3h as backup in case automatic updates are not working
scheduler.start()