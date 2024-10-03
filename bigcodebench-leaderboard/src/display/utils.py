from dataclasses import dataclass, make_dataclass
from enum import Enum
import json
import logging
from datetime import datetime
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Convert ISO 8601 dates to datetime objects for comparison
def parse_iso8601_datetime(date_str):
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+00:00'
    return datetime.fromisoformat(date_str)

def parse_datetime(datetime_str):
    formats = [
        "%Y-%m-%dT%H-%M-%S.%f",  # Format with dashes
        "%Y-%m-%dT%H:%M:%S.%f",  # Standard format with colons
        "%Y-%m-%dT%H %M %S.%f",  # Spaces as separator
    ]

    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    # in rare cases set unix start time for files with incorrect time (legacy files)
    logging.error(f"No valid date format found for: {datetime_str}")
    return datetime(1970, 1, 1)


def load_json_data(file_path):
    """Safely load JSON data from a file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Error reading JSON from {file_path}")
        return None  # Or raise an exception


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


column_map = {
    "T": "T",
    "model": "Model",
    "type": "Model Type",
    "size_range": "Size Range",
    "complete": "Complete",
    "instruct": "Instruct",
    "average": "Average",
    "elo_mle": "Elo Rating",
    "link": "Link",
    "act_param": "#Act Params (B)",
    "size": "#Params (B)",
    "moe": "MoE",
    # "lazy": "Lazy",
    "openness": "Openness",
    # "direct_complete": "Direct Completion",
}

type_map = {
    "🔶": "🔶 Chat Models (RLHF, DPO, IFT, ...)",
    "🟢": "🟢 Base Models"
}

moe_map = {
    True: "MoE",
    False: "Dense"
}
# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass(frozen=True)
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False
    dummy: bool = False


auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(["T", ColumnContent, ColumnContent(column_map["T"], "str", True, never_hidden=True)])
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent(column_map["model"], "markdown", True, never_hidden=True)])
auto_eval_column_dict.append(["type", ColumnContent, ColumnContent(column_map["type"], "str", False, True)])
auto_eval_column_dict.append(["size_range", ColumnContent, ColumnContent(column_map["size_range"], "str", False, True)])
# Scores
auto_eval_column_dict.append(["complete", ColumnContent, ColumnContent(column_map["complete"], "number", True)])
auto_eval_column_dict.append(["instruct", ColumnContent, ColumnContent(column_map["instruct"], "number", True)])
auto_eval_column_dict.append(["average", ColumnContent, ColumnContent(column_map["average"], "number", True)])
auto_eval_column_dict.append(["elo_mle", ColumnContent, ColumnContent(column_map["elo_mle"], "number", True)])

# Model information
auto_eval_column_dict.append(["act_param", ColumnContent, ColumnContent(column_map["act_param"], "number", True)])
auto_eval_column_dict.append(["link", ColumnContent, ColumnContent(column_map["link"], "str", False, True)])
auto_eval_column_dict.append(["size", ColumnContent, ColumnContent(column_map["size"], "number", False)])
# auto_eval_column_dict.append(["lazy", ColumnContent, ColumnContent(column_map["lazy"], "bool", False, True)])
auto_eval_column_dict.append(["moe", ColumnContent, ColumnContent(column_map["moe"], "str", False, True)])
auto_eval_column_dict.append(["openness", ColumnContent, ColumnContent(column_map["openness"], "str", False, True)])
# auto_eval_column_dict.append(["direct_complete", ColumnContent, ColumnContent(column_map["direct_complete"], "bool", False)])

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model_link = ColumnContent("link", "markdown", True)
    model_name = ColumnContent("model", "str", True)

@dataclass
class ModelDetails:
    name: str
    symbol: str = ""  # emoji, only for the model type


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn)]
TYPES = [c.type for c in fields(AutoEvalColumn)]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]


NUMERIC_INTERVALS = {
    "?": pd.Interval(-1, 0, closed="right"),
    "~1.5": pd.Interval(0, 2, closed="right"),
    "~3": pd.Interval(2, 4, closed="right"),
    "~7": pd.Interval(4, 9, closed="right"),
    "~13": pd.Interval(9, 20, closed="right"),
    "~35": pd.Interval(20, 45, closed="right"),
    "~60": pd.Interval(45, 70, closed="right"),
    "70+": pd.Interval(70, 10000, closed="right"),
}
