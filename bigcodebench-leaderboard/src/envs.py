import os
from huggingface_hub import HfApi

# clone / pull the lmeh eval data
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DATA_VERSION = "v0.1.1"

REPO_ID = "bigcode/bigcodebench-leaderboard"
QUEUE_REPO = "bigcode/bigcodebench-requests"
DATA_REPO = "bigcode/bigcodebench"
RESULT_REPO = "bigcode/bigcodebench-results"
HARD_RESULT_REPO = "bigcode/bigcodebench-hard-results"

ELO_REPO = "bigcode/bigcodebench-elo"
HARD_ELO_REPO = "bigcode/bigcodebench-hard-elo"
SOLVE_REPO = "bigcode/bigcodebench-solve-rate"
HARD_SOLVE_REPO = "bigcode/bigcodebench-hard-solve-rate"

VOTES_REPO = "bigcode/bigcodebench-votes"

HF_HOME = os.getenv("HF_HOME", ".")

# Check HF_HOME write access
print(f"Initial HF_HOME set to: {HF_HOME}")

if not os.access(HF_HOME, os.W_OK):
    print(f"No write access to HF_HOME: {HF_HOME}. Resetting to current directory.")
    HF_HOME = "."
    os.environ["HF_HOME"] = HF_HOME
else:
    print("Write access confirmed for HF_HOME")

VOTES_PATH = os.path.join(HF_HOME, "model-votes")
EVAL_REQUESTS_PATH = os.path.join(HF_HOME, "eval-queue")

# Rate limit variables
RATE_LIMIT_PERIOD = 7
RATE_LIMIT_QUOTA = 5
HAS_HIGHER_RATE_LIMIT = []

API = HfApi(token=HF_TOKEN)
