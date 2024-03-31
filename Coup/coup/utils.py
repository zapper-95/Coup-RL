# Help from GPT-4 writing this.

from glob import glob
import os
import ray
from ray.rllib.algorithms.algorithm import Algorithm


def get_best_agent():
    """Gets the agent that performed the best against random agents."""
    raise NotImplementedError("This function is not implemented yet.")


def get_experiment_folder(alg_name="PPO"):
    # Define the path to the results folder
    results_folder_path = os.path.abspath("./ray_results/{}".format(alg_name))

    # Corrected: Filter only directories for experiments
    experiment_folders = [f for f in glob(os.path.join(results_folder_path, "{}_*".format(alg_name))) if os.path.isdir(f)]
    #print(f"Experiment folders found: {experiment_folders}")
    return experiment_folders


def get_checkpoints_folder(latest_experiment, alg_name="PPO"):
    # Corrected: More accurately find checkpoint directories within the latest experiment folder
    checkpoint_dirs = glob(os.path.join(latest_experiment, "checkpoint_*"), recursive=True)
    print(f"Checkpoint directories found: {checkpoint_dirs}")

    if not checkpoint_dirs:
        raise ValueError("No checkpoints found in the latest experiment directory.")
    return checkpoint_dirs

def get_last_agent_path(alg_name="PPO"):
    """Gets the path to the most recent folder and checkpoint for a given algorithm."""

    experiment_folders = get_experiment_folder(alg_name)

    # Find the most recent experiment folder
    latest_experiment = max(experiment_folders, key=os.path.getmtime)
    print(f"Latest experiment folder: {latest_experiment}")

    checkpoint_dirs = get_checkpoints_folder(latest_experiment, alg_name)

    # Corrected: Find the most recent checkpoint based on checkpoint number
    latest_checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(os.path.basename(x).split('_')[-1]))
    print(f"Latest checkpoint directory: {latest_checkpoint_dir}")

    return latest_checkpoint_dir


def get_penultimate_agent_path(alg_name="PPO"):
    """Gets the path to the second most recent folder and checkpoint for a given algorithm."""


    experiment_folders = get_experiment_folder(alg_name)

    # Find the second most recent experiment folder
    second_latest = sorted(experiment_folders, key=os.path.getmtime)[-2]
    print(f"Second latest experiment folder: {second_latest}")

    checkpoint_dirs = get_checkpoints_folder(second_latest, alg_name)
    
    # Find the most recent checkpoint based on checkpoint number

    second_checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(os.path.basename(x).split('_')[-1]))
    print(f"Second latest checkpoint directory: {second_checkpoint_dir}")



    return second_checkpoint_dir


def get_nth_latest_model(exp_n=1, check_n=1, alg_name="PPO"):
    experiment_folders = get_experiment_folder(alg_name)

    # Find the second most recent experiment folder
    experiment = sorted(experiment_folders, key=os.path.getmtime)[-exp_n]
    print(f"{exp_n}th last experiment folder: {experiment}")

    checkpoint_dirs = get_checkpoints_folder(experiment, alg_name)
    
    # Find the most recent checkpoint based on checkpoint number

    checkpoint = sorted(checkpoint_dirs, key=lambda x: int(os.path.basename(x).split('_')[-1]))[-check_n]
    print(f"{check_n}th last checkpoint directory: {checkpoint}")

    return checkpoint