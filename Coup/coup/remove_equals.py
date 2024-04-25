import os
for root, dirs, files in os.walk("./ray_results/PPO_decentralised"):
    for dir in dirs:
        new_dir = dir.replace("=", "-")
        new_dir = dir.replace(",", "-")
        os.rename(os.path.join(root, dir), os.path.join(root, new_dir))



def fix_dir_names(directory):

    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            new_dir = dir.replace("=", "-")
            new_dir = dir.replace(",", "-")
            os.rename(os.path.join(root, dir), os.path.join(root, new_dir))