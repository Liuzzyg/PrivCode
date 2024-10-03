import json
import logging
import pathlib
import pandas as pd
import gradio as gr
import schedule
import time
from datetime import datetime, timezone

from src.envs import API

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoteManager:
    def __init__(self, votes_path, eval_requests_path, repo_id):
        self.votes_path = votes_path
        self.eval_requests_path = eval_requests_path
        self.repo_id = repo_id
        self.vote_dataset = self.read_vote_dataset()
        self.vote_check_set = self.make_check_set(self.vote_dataset)
        self.votes_to_upload = []

    def init_vote_dataset(self):
        self.vote_dataset = self.read_vote_dataset()
        self.vote_check_set = self.make_check_set(self.vote_dataset)

    def read_vote_dataset(self):
        result = []
        votes_file = pathlib.Path(self.votes_path) / "votes_data.jsonl"
        if votes_file.exists():
            with open(votes_file, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    result.append(data)
        result = pd.DataFrame(result)
        return result

    def make_check_set(self, vote_dataset: pd.DataFrame):
        result = list()
        for row in vote_dataset.itertuples(index=False, name='vote'):
            result.append((row.model, row.revision, row.username))
        return set(result)
    
    def get_model_revision(self, selected_model: str) -> str:
        """Fetch the revision for the given model from the request files."""
        for user_folder in pathlib.Path(self.eval_requests_path).iterdir():
            if user_folder.is_dir():
                for file in user_folder.glob("*.json"):
                    with open(file, "r") as f:
                        data = json.load(f)
                        if data.get("model") == selected_model:
                            return data.get("revision", "main")
        return "main"

    def create_request_vote_df(self, pending_models_df: gr.Dataframe):
        if pending_models_df.empty or not "model_name" in pending_models_df.columns:
            return pending_models_df
        self.vote_dataset = self.read_vote_dataset()
        vote_counts = self.vote_dataset.groupby(['model', 'revision']).size().reset_index(name='vote_count')

        pending_models_df_votes = pd.merge(
            pending_models_df, 
            vote_counts, 
            left_on=["model_name", 'revision'], 
            right_on=['model', 'revision'], 
            how='left'
        )
        # Filling empty votes
        pending_models_df_votes['vote_count'] = pending_models_df_votes['vote_count'].fillna(0)
        pending_models_df_votes = pending_models_df_votes.sort_values(by=["vote_count", "model_name"], ascending=[False, True])
        # Removing useless columns
        pending_models_df_votes = pending_models_df_votes.drop(["model_name", "model"], axis=1)
        return pending_models_df_votes

    # Function to be called when a user votes for a model
    def add_vote(
            self,
            selected_model: str,
            pending_models_df: gr.Dataframe,
            profile: gr.OAuthProfile | None
        ):
        logger.debug(f"Type of list before usage: {type(list)}")
        # model_name, revision, user_id, timestamp
        if selected_model in ["str", ""]:
            gr.Warning("No model selected")
            return
        
        if profile is None:
            gr.Warning("Hub Login required")
            return

        vote_username = profile.username
        model_revision = self.get_model_revision(selected_model)
        
        # tuple (immutable) for checking than already voted for model
        check_tuple = (selected_model, model_revision, vote_username)
        if check_tuple in self.vote_check_set:
            gr.Warning("Already voted for this model")
            return
        
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        vote_obj = {
            "model": selected_model,
            "revision": model_revision,
            "username": vote_username,
            "timestamp": current_time
        }

        # Append the vote to the JSONL file
        try:
            votes_file = pathlib.Path(self.votes_path) / "votes_data.jsonl"
            with open(votes_file, "a") as f:
                f.write(json.dumps(vote_obj) + "\n")
            logger.info(f"Vote added locally: {vote_obj}")

            self.votes_to_upload.append(vote_obj)
        except Exception as e:
            logger.error(f"Failed to write vote to file: {e}")
            gr.Warning("Failed to record vote. Please try again")
            return
        
        self.vote_check_set.add(check_tuple)
        gr.Info(f"Voted for {selected_model}")

        return self.create_request_vote_df(pending_models_df)

    def upload_votes(self):
        if self.votes_to_upload:
            votes_file = pathlib.Path(self.votes_path) / "votes_data.jsonl"
            try:
                with open(votes_file, "rb") as f:
                    API.upload_file(
                        path_or_fileobj=f,
                        path_in_repo="votes_data.jsonl",
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        commit_message="Updating votes_data.jsonl with new votes",
                    )
                logger.info("Votes uploaded to votes repository")
                self.votes_to_upload.clear()
            except Exception as e:
                logger.error(f"Failed to upload votes to repository: {e}")

def run_scheduler(vote_manager):
    while True:
        schedule.run_pending()
        time.sleep(1)
