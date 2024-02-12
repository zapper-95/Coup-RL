from pettingzoo.utils import wrappers
from stable_baselines3.common.env_checker import check_env
from coup import CoupEnv as env

# It will check your custom environment and output additional warnings if needed


env = wrappers.SingleAgentWrapper(env)

check_env(env)