import pathlib
import pandas as pd
from datasets import Dataset
from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.display.utils import load_json_data, column_map, type_map, moe_map, NUMERIC_INTERVALS



def get_evaluation_queue_df(save_path, cols):
    """Generate dataframes for pending, running, and finished evaluation entries."""
    save_path = pathlib.Path(save_path)
    all_evals = []

    for path in save_path.rglob("*.json"):
        data = load_json_data(path)
    # Organizing data by status
    status_map = {
        "PENDING": ["PENDING", "RERUN"],
        "RUNNING": ["RUNNING"],
        "FINISHED": ["FINISHED", "PENDING_NEW_EVAL"],
    }
    status_dfs = {status: [] for status in status_map}
    for eval_data in all_evals:
        for status, extra_statuses in status_map.items():
            if eval_data["status"] in extra_statuses:
                status_dfs[status].append(eval_data)

    return tuple(pd.DataFrame(status_dfs[status], columns=cols) for status in ["FINISHED", "RUNNING", "PENDING"])


def get_leaderboard_df(leaderboard_dataset: Dataset, cols: list):
    """Retrieve and process leaderboard data."""
    all_data_json = leaderboard_dataset.to_dict()
    num_items = leaderboard_dataset.num_rows
    all_data_json_list = [{k: all_data_json[k][ix] for k in all_data_json.keys()} for ix in range(num_items)]

    df = pd.DataFrame.from_records(all_data_json_list)
    # replace df.moe true to false, false to true
    # map column names
    df = df.rename(columns=column_map)
    df[AutoEvalColumn.moe.name] = df[AutoEvalColumn.moe.name].map(moe_map)
    df[AutoEvalColumn.T.name] = df[AutoEvalColumn.type.name]
    df[AutoEvalColumn.type.name] = df[AutoEvalColumn.type.name].map(type_map)
    df[AutoEvalColumn.average.name] = df.apply(lambda x: round((x[AutoEvalColumn.complete.name] + x[AutoEvalColumn.instruct.name]) / 2, 1) if not pd.isna(x[AutoEvalColumn.complete.name]) and not pd.isna(x[AutoEvalColumn.instruct.name]) else None, axis=1)
    df[AutoEvalColumn.size_range.name] = df[AutoEvalColumn.size.name].apply(lambda x: next((k for k, v in NUMERIC_INTERVALS.items() if x in v), "?"))
    df = make_clickable_model(df, AutoEvalColumn.model.name, AutoEvalColumn.link.name)
    df = df.sort_values(by=[AutoEvalColumn.complete.name], ascending=False)
    df = df[cols].round(decimals=2)
    return df