from glob import glob
import os

def get_experiment_folders(base_path):
    # Define the path to the results folder
    results_folder_path = os.path.abspath(base_path)

    # Corrected: Filter only directories for experiments
    experiment_folders = [f for f in glob(os.path.join(results_folder_path, "*")) if os.path.isdir(f)]
    #print(f"Experiment folders found: {experiment_folders}")
    return experiment_folders


def get_checkpoints_folder(experiment_folder):
    """Retrieve checkpoint folders from an experiment directory."""
    checkpoint_dirs = glob(os.path.join(experiment_folder, "checkpoint_*"))
    if not checkpoint_dirs:
        raise ValueError("No checkpoints found in the experiment directory.")
    return checkpoint_dirs

def get_sorted_checkpoints(checkpoint_dirs):
    """Sort checkpoint directories by the checkpoint number."""
    return sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[-1]))

def get_latest_folder(folders):
    """Retrieve the latest folder based on modification time."""
    if not folders:
        raise ValueError("No experiment folders found.")
    return max(folders, key=os.path.getmtime)

def get_nth_folder(folders, n=1):
    """Retrieve the nth latest folder based on modification time."""
    if len(folders) < n:
        raise ValueError(f"Requested the {n}th latest folder, but only {len(folders)} folders exist.")
    return sorted(folders, key=os.path.getmtime, reverse=True)[n-1]


def get_last_agent_path(critic_type="centralised", n=1):
    base_path = os.path.abspath(f"./ray_results/PPO_{critic_type}")
    experiment_folders = get_experiment_folders(base_path)

    print(experiment_folders)

    if not experiment_folders:
        raise ValueError("No experiment folders found. Check the directory and naming convention.")
    
    if n == 1:
        latest_folder = get_latest_folder(experiment_folders)
    else:
        latest_folder = get_nth_folder(experiment_folders, n)
    
    print("selected folder ", latest_folder)
    checkpoint_dirs = get_checkpoints_folder(latest_folder)
    latest_checkpoint = get_sorted_checkpoints(checkpoint_dirs)[-1]
    

    return latest_checkpoint

# def main(critic_type="centralised", n=1):
#     base_path = os.path.abspath(f"./ray_results/PPO_{critic_type}")
#     experiment_folders = get_experiment_folders(base_path, critic_type)

#     if not experiment_folders:
#         raise ValueError("No experiment folders found. Check the directory and naming convention.")
    
#     if n == 1:
#         latest_folder = get_latest_folder(experiment_folders)
#     else:
#         latest_folder = get_nth_folder(experiment_folders, n)
    
#     checkpoint_dirs = get_checkpoints_folder(latest_folder)
#     latest_checkpoint = get_sorted_checkpoints(checkpoint_dirs)[-1]
    

#     return latest_checkpoint

# if __name__ == "__main__":
#     main()
