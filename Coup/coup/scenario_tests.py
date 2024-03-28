import coup_v1
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from train_ppo import ActionMaskModel
from utils import get_last_agent_path, get_penultimate_agent_path
import ray
import numpy as np


"""Code based on https://github.com/BStarcheus/open_spiel_coup/blob/master/coup_experiments/scripts/policy_analysis.py"""


model_path = get_last_agent_path()


def env_creator():
    env = coup_v1.env(render_mode="human")
    return env


ModelCatalog.register_custom_model("am_model", ActionMaskModel)
register_env("Coup", lambda config: PettingZooEnv(env_creator()))

policy = Algorithm.from_checkpoint(model_path).get_policy(policy_id="player_1")

def get_probs_of_action(env, action, action_mask, policy, extra_fetches):
    
    # get the model that processes observations
    model = policy.model

    # get the logits of the action distribution
    dist_inputs = extra_fetches['action_dist_inputs']

    # get the action distribution class
    dist_class = policy.dist_class

    # create the action distribution using the logits
    action_dist = dist_class(dist_inputs, model)

    probs = action_dist.dist.probs.tolist()

    for i in range(len(action_mask)):
        if action_mask[i] == 1 and env.get_action_string(i) == action:
            return probs[i]
    return None

def foreign_aid_test():
    '''
    Test FA prob before and after being blocked by Duke.
    Should see a decrease in prob.
    '''

    prob_fa = []
    
    

    env = env_creator()
    env.reset()


    actions = [env.get_action_id("foreign_aid"), env.get_action_id("counteract"), env.get_action_id("pass"), env.get_action_id("tax")]


    env.state_space["player_1_card_1"] = "ambassador"
    env.state_space["player_1_card_2"] = "contessa"  

    env.state_space["player_2_card_1"] = "duke"
    env.state_space["player_2_card_2"] = "duke"
    
    i = 0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        _, action_mask = obs.values()

        if agent == env.possible_agents[0]:
            _, _, extra_fetches = policy.compute_single_action(obs)

            prob = get_probs_of_action(env, "foreign_aid", action_mask, policy, extra_fetches)

            # only add the probability, if the action was available
            if prob is not None:
                prob_fa.append(prob)       

        if i < len(actions):
            env.step(actions[i])
        else:
            break

        i += 1
    
    print(prob_fa)
    assert len(prob_fa) == 2, "Should have 2 probabilities"
    assert prob_fa[0] > prob_fa[1], "Probabilities should decrease after Duke blocks FA"
    print("Foreign Aid test passed")

    



def steal_test():
    '''
    Test Steal prob before and after being blocked.
    Should see a decrease in prob.
    '''

    prob_steal = []
    
    

    env = env_creator()
    env.reset()


    actions = [env.get_action_id("steal"), env.get_action_id("counteract"), env.get_action_id("pass"), env.get_action_id("income")]


    env.state_space["player_1_card_1"] = "captain"
    env.state_space["player_1_card_2"] = "captain"  

    env.state_space["player_2_card_1"] = "ambassador"
    env.state_space["player_2_card_1"] = "ambassador"
    
    i = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        _, action_mask = obs.values()

        if agent == env.possible_agents[0]:
            _, _, extra_fetches = policy.compute_single_action(obs)

            prob = get_probs_of_action(env, "steal", action_mask, policy, extra_fetches)

            # only add the probability, if the action was available
            if prob is not None:
                prob_steal.append(prob)       

        if i < len(actions):
            env.step(actions[i])
        else:
            break

        i += 1
    
    print(prob_steal)
    assert len(prob_steal) == 2, "Should have 2 probabilities"
    assert prob_steal[0] > prob_steal[1], "Probabilities should decrease after Ambassador blocks Steak"
    print("Steal test passed")



def counter_assassinate_test():
    """
    A player with one card remaining which is the contessa, if assassinated should use the contessa to counteract, otherwise
    they risk losing.
    
    """

    prob_counter = []
    prob_challenge = []

    env = env_creator()
    env.reset()


    actions = [
        env.get_action_id("income"), env.get_action_id("income"),
        env.get_action_id("income"), env.get_action_id("assassinate"),
        env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("income"),
        env.get_action_id("income"), env.get_action_id("income"),         
        env.get_action_id("income"), env.get_action_id("income"),         
        env.get_action_id("income"), env.get_action_id("assassinate"),         
        ]    


    env.state_space["player_1_card_1"] = "contessa"
    env.state_space["player_1_card_2"] = "contessa"  

    env.state_space["player_2_card_1"] = "assassin"
    env.state_space["player_2_card_1"] = "assassin"




    i = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        _, action_mask = obs.values()

        if agent == env.possible_agents[0]:
            _, _, extra_fetches = policy.compute_single_action(obs)

            prob_co = get_probs_of_action(env, "counteract", action_mask, policy, extra_fetches)
            prob_ch = get_probs_of_action(env, "challenge", action_mask, policy, extra_fetches)
            # only add the probability, if the action was available
            if prob_co is not None:
                prob_counter.append(prob_co)

            if prob_ch is not None:
                prob_challenge.append(prob_ch)       



        if i < len(actions):
            env.step(actions[i])
        else:
            break

        i += 1


    assert len(prob_counter) == 2 , "Should have 2 probabilities for counteracting"
    assert len(prob_challenge) == 2 , "Should have 2 probabilities for challenging"
    
    print(f"prob_counter {prob_counter}")
    print(f"prob_challenge {prob_challenge}")

    assert prob_counter[1] > prob_counter[0], "Probability of counteracting should increase with only one card remaining"
    assert prob_challenge[1] > prob_challenge[0], "Probability of challenging should increase with only one card remaining"
    
    assert prob_counter[1] > prob_challenge[1], "Probability of counteracting should be higher than challenging, as unnecessary risk to challenge with only one card remaining"
    print("Counter Assassinate test passed")
foreign_aid_test()
steal_test()
counter_assassinate_test()
