import coup_v2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel as ActionMaskModel
from utils import get_last_agent_path
import ray
import numpy as np


"""Code based on https://github.com/BStarcheus/open_spiel_coup/blob/master/coup_experiments/scripts/policy_analysis.py"""



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

    if action == "any":
        return probs

    for i in range(len(action_mask)):
        if action_mask[i] == 1 and env.get_action_string(i) == action:
            return probs[i]
    return None

def foreign_aid_test(policy, env):
    '''
    Test FA prob before and after being blocked by Duke.
    Should see a decrease in prob.
    '''

    prob_fa = []
    
    

    env.reset()


    actions = [env.get_action_id("foreign_aid"), env.get_action_id("counteract"), env.get_action_id("pass"), env.get_action_id("tax"), env.get_action_id("pass"), env.get_action_id("pass")]


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

    if len(prob_fa) != 2:
        print("Should have 2 probabilities")
        return False

    if prob_fa[0] < prob_fa[1]:
        print("Probabilities should decrease after Duke blocks FA")
        return False

    print("Foreign Aid test passed")
    return True

    


def coup_test(policy, env):
    '''
    Test Coup prob for 7-9 coins.
    '''
    probs = []
    probs_coup = []

    env.reset()


    actions = [
            #3, 2
            env.get_action_id("foreign_aid"),
            env.get_action_id("pass"), 
            env.get_action_id("pass"),

            # 3, 3
            env.get_action_id("income"),
            env.get_action_id("pass"), 
            env.get_action_id("pass"),

            # 5, 3
            env.get_action_id("foreign_aid"),
            env.get_action_id("pass"), 
            env.get_action_id("pass"),


            # 5, 4
            env.get_action_id("income"),
            env.get_action_id("pass"), 
            env.get_action_id("pass"),

            # 7, 4
            env.get_action_id("foreign_aid"),
            env.get_action_id("pass"),
            env.get_action_id("pass"), 
            
            # 7, 5
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),           
            
            # 8, 5
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),

            # 8, 6
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),

            # 9, 6
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),


            # 9, 7
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),
            ]


    env.state_space["player_1_card_1"] = "ambassador"
    env.state_space["player_1_card_2"] = "ambassador"  

    env.state_space["player_2_card_1"] = "contessa"
    env.state_space["player_2_card_2"] = "contessa"
    
    i = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        _, action_mask = obs.values()

        if agent == env.possible_agents[0]:
            _, _, extra_fetches = policy.compute_single_action(obs)

            prob = get_probs_of_action(env, "any", action_mask, policy, extra_fetches)
            prob_coup = get_probs_of_action(env, "coup", action_mask, policy, extra_fetches)
            # only add the probability, if the action was available
            if prob_coup is not None:
                probs.append(prob)
                probs_coup.append(prob_coup)

        if i < len(actions):
            env.step(actions[i])
        else:
            break

        i += 1
    
    if len(probs) != 3:
        print("Should have 3 probabilities")
        return False
    print(probs, probs_coup)
    for i in range(3):
        if max(probs[i]) != probs_coup[i]:
            print("Coup should have the highest probability")
            return False

    print("Coup test passed")
    return True

def assassinate_test(policy, env):
    '''
    See prob of assassinate with/without card in hand.
    '''
    probs_assa = [[],[]]

    env.reset()


    actions = [
            env.get_action_id("income"),
            env.get_action_id("pass"), 
            env.get_action_id("pass"),

            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"), 
            
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),           
            
            env.get_action_id("income"),
            env.get_action_id("pass"),
            env.get_action_id("pass"),
            ]

    player_1_cards = [["ambassador", "ambassador"], ["assassin", "assassin"]]  
    player_2_cards = [["captain", "captain"], ["ambassador", "ambassador"]]

    for j in range(2):
        env.reset()  
        env.state_space["player_1_card_1"] = player_1_cards[j][0]
        env.state_space["player_1_card_2"] = player_1_cards[j][1]

        env.state_space["player_2_card_1"] = player_2_cards[j][0]
        env.state_space["player_2_card_2"] = player_2_cards[j][1]

        
        i = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            _, action_mask = obs.values()

            if agent == env.possible_agents[0]:
                _, _, extra_fetches = policy.compute_single_action(obs)

                prob = get_probs_of_action(env, "assassinate", action_mask, policy, extra_fetches)
                # only add the probability, if the action was available
                if prob is not None:
                    probs_assa[j].append(prob)

            if i < len(actions):
                env.step(actions[i])
            else:
                break

            i += 1
    
    if len(probs_assa[0]) != 1 or len(probs_assa[1]) != 1:
        print("Should have 1 probability")
        return False

    print(probs_assa[0], probs_assa[1])

    if probs_assa[0][0] > probs_assa[1][0]:
        print("Should have higher probability of assassinate with card in hand")
        return False

    print("Assassinate test passed")
    return True

def steal_test(policy, env):
    '''
    Test Steal prob before and after being blocked.
    Should see a decrease in prob.
    '''

    prob_steal = []
    
    


    env.reset()


    actions = [env.get_action_id("steal")
               ,env.get_action_id("counteract")
               ,env.get_action_id("pass")
               , env.get_action_id("income"),
                 env.get_action_id("pass"),
                 env.get_action_id("pass")]


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

    if len(prob_steal) != 2:
        print("Should have 2 probabilities")
        return False

    if prob_steal[0] < prob_steal[1]:
        print("Probabilities should decrease after Ambassador blocks Steal")
        return False

    print("Steal test passed")
    return True


def counter_assassinate_test(policy, env):
    '''
    See prob of block or challenge an asassination.
    With 2 cards remaining, should not risk double elimination.
    With 1 card remaining, nothing to lose.
    '''

    prob_counter = []
    prob_challenge = []

    env.reset()


    actions = [
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("assassinate"), env.get_action_id("kill_card_1"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"),         
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"),          
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"),
        env.get_action_id("assassinate"),         
        ]    


    env.state_space["player_1_card_1"] = "ambassador"
    env.state_space["player_1_card_2"] = "ambassador"  

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


    if len(prob_counter) != 2:
        print("Should have 2 probabilities for counteracting")
        return False
    
    if len(prob_challenge) != 2:
        print("Should have 2 probabilities for challenging")
        return False
    
    print(f"prob_counter {prob_counter}")
    print(f"prob_challenge {prob_challenge}")

    if prob_counter[0] > prob_counter[1]:
        print("Probability of counteracting should increase with only one card remaining")
        return False
    
    if prob_challenge[0] > prob_challenge[1]:
        print("Probability of challenging should increase with only one card remaining")
        return False

    print("Counter Assassinate test passed")
    return True


def counter_assassinate_test_2(policy, env):
    '''
    The probability of counteracting or challenging an assassination with one card remaining should sum to be close to 1, as a player has nothing to lose.
    Otherwise, they essentially admit defeat

    '''

    prob_counter = []
    prob_challenge = []

    env.reset()


    actions = [
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("assassinate"), env.get_action_id("kill_card_1"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"),         
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"),          
        env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"),
        env.get_action_id("assassinate"),         
        ]    


    env.state_space["player_1_card_1"] = "ambassador"
    env.state_space["player_1_card_2"] = "ambassador"  

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


    if len(prob_counter) != 2:
        print("Should have 2 probabilities for counteracting")
        return False
    
    if len(prob_challenge) != 2:
        print("Should have 2 probabilities for challenging")
        return False
    
    print(f"prob_counter {prob_counter}")
    print(f"prob_challenge {prob_challenge}")

    if prob_counter[1] + prob_challenge[1] < 0.99:
        print("Probability of counteracting or challenging should sum to be close to 1 when a player only has one card remaining.")
        return False

    print("Counter Assassinate test 2 passed")
    return True

# def contessa_test():
#     """A player with contessa should with higher probability counteract an assassination. This should increase when they have only one card remaining."""

#     prob_counter = []
#     prob_challenge = []

#     env = env_creator()
#     env.reset()


#     actions = [
#         env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("assassinate"), env.get_action_id("kill_card_1"), env.get_action_id("pass"), 
#         env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("income"), env.get_action_id("pass"), env.get_action_id("pass"),         
#         env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"), 
#         env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"),          
#         env.get_action_id("income"),env.get_action_id("pass"), env.get_action_id("pass"),
#         env.get_action_id("assassinate"),         
#         ]    


#     env.state_space["player_1_card_1"] = "contessa"
#     env.state_space["player_1_card_2"] = "contessa"  

#     env.state_space["player_2_card_1"] = "assassin"
#     env.state_space["player_2_card_1"] = "assassin"




#     i = 0
#     for agent in env.agent_iter():
#         obs, reward, termination, truncation, info = env.last()
#         _, action_mask = obs.values()

#         if agent == env.possible_agents[0]:
#             _, _, extra_fetches = policy.compute_single_action(obs)

#             prob_co = get_probs_of_action(env, "counteract", action_mask, policy, extra_fetches)
#             prob_ch = get_probs_of_action(env, "challenge", action_mask, policy, extra_fetches)
#             # only add the probability, if the action was available
#             if prob_co is not None:
#                 prob_counter.append(prob_co)

#             if prob_ch is not None:
#                 prob_challenge.append(prob_ch)       



#         if i < len(actions):
#             env.step(actions[i])
#         else:
#             break

#         i += 1


#     if len(prob_counter) != 2:
#         print("Should have 2 probabilities for counteracting")
#         return False

#     if len(prob_challenge) != 2:
#         print("Should have 2 probabilities for challenging")
#         return False

    
#     print(f"prob_counter {prob_counter}")
#     print(f"prob_challenge {prob_challenge}")


#     if prob_counter[0] > prob_counter[1]:
#         print("Probability of counteracting should increase with only one card remaining")
#         return False
    

#     if prob_counter[1] < prob_challenge[1]:
#         print("Probability of counteracting should be higher than challenging, as unnecessary risk to challenge with only one card remaining")
#         return False


#     print("Counter Assassinate test passed")
#     return True

# def get_test_results():
#     def env_creator():
#         env = coup_v2.env()
#         return env


#     ray.init(local_mode=True)
#     ModelCatalog.register_custom_model("am_model", ActionMaskModel)
#     register_env("Coup", lambda config: PettingZooEnv(env_creator()))

#     policy = Algorithm.from_checkpoint(model_path).get_policy(policy_id="policy")
#     return foreign_aid_test(), steal_test(), counter_assassinate_test(), coup_test(), assassinate_test()


# print(
# foreign_aid_test()+
# steal_test()+
# counter_assassinate_test()+
# coup_test()+
# assassinate_test())
