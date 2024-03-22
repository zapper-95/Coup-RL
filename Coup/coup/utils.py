# Help from GPT-4 writing this.

from glob import glob
import os
import ray
from ray.rllib.algorithms.algorithm import Algorithm


def get_best_agent():
    """Gets the agent that performed the best against random agents."""
    raise NotImplementedError("This function is not implemented yet.")


def get_last_agent_path(alg_name="PPO"):
    """Gets the path to the most recent folder and checkpoint for a given algorithm."""


    # Define the path to the results folder
    results_folder_path = os.path.abspath("./ray_results/{}".format(alg_name))

    # Corrected: Filter only directories for experiments
    experiment_folders = [f for f in glob(os.path.join(results_folder_path, "{}_*".format(alg_name))) if os.path.isdir(f)]
    print(f"Experiment folders found: {experiment_folders}")

    # Find the most recent experiment folder
    latest_experiment = max(experiment_folders, key=os.path.getmtime)
    print(f"Latest experiment folder: {latest_experiment}")

    # Corrected: More accurately find checkpoint directories within the latest experiment folder
    checkpoint_dirs = glob(os.path.join(latest_experiment, "checkpoint_*"), recursive=True)
    print(f"Checkpoint directories found: {checkpoint_dirs}")

    if not checkpoint_dirs:
        raise ValueError("No checkpoints found in the latest experiment directory.")

    # Corrected: Find the most recent checkpoint based on checkpoint number
    latest_checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(os.path.basename(x).split('_')[-1]))
    print(f"Latest checkpoint directory: {latest_checkpoint_dir}")



    return latest_checkpoint_dir