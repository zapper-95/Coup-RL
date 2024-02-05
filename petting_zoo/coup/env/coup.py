from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
#from player import Player
from deck import Deck
from gymnasium.spaces import Discrete
import gymnasium
import numpy as np


# based on code from https://pettingzoo.farama.org/content/environment_creation/



ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
CARDS = ["ambassador", "assassin", "captain", "contessa", "duke"]
ACTIONS = [
    "income",
    "foreign_aid",
    "tax",
    "assassinate",
    "exchange",
    "steal",
    "counteract",
    "challenge",
    "coup"
]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}



def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = CoupEnv(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env



class CoupEnv(AECEnv):
    metadata = {
        "name": "coup",
    }

    def __init__(self, render_mode=None):
        
        
        self.render_mode = render_mode
        
        self.state_space = {
            "player_1_card_1": None,
            "player_1_card_2": None,
            "player_2_card_1": None,
            "player_2_card_2": None,
            "player_1_card_1_alive": True,
            "player_1_card_2_alive": True,
            "player_2_card_1_alive": True,
            "player_2_card_2_alive": True,
            "player_1_coins": 1,
            "player_2_coins": 1,
            "player_1_action": None,
            "player_2_action": None
        }

        self.player_turn = 0
        self.agents = [f"player_{i+1}" for i in range(2)]

        # dictionary of the agent names to their index
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.agents))))
        )
        self.deck = Deck(CARDS)


        self._action_spaces = {agent: Discrete(9) for agent in self.agents}
        
        # the players can see all of the state, but other players hands
        self._observation_spaces = {
            agent: Discrete(10) for agent in self.agents
        }
    

    def get_next_agent(self): 
        return self._agent_selector.next() 

    def increment_next_agent(self):
        self.agent_selection = self._agent_selector.next()
        

    def get_action_string(self, action_id):
        if action_id is None:
            return None
        
        return ACTIONS[action_id]

    def update_state(self, actions):
        """Updates the state space of the environment"""
        if "player_1" in actions:
            self.state_space["player_1_action"] = actions["player_1"]
        else:
            self.state_space["player_2_action"] = actions["player_2"]

    def remove_actions(self):
        self.state_space["player_1_action"] = None
        self.state_space["player_2_action"] = None
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]


    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        """Print the state space in a readable format"""
        print("----------------")
        print(f"Turn: P{self.player_turn+1}")
        alive_cards = []

        print("----------------")
        if self.player_turn == 0:
            print(f"Action: {self.state_space['player_1_action']}")
        else:
            print(f"Action: {self.state_space['player_2_action']}")
        print("----------------")


        for i in range(2):
            for j in range(2):
                if self.state_space[f"player_{i+1}_card_{j+1}_alive"]:
                    alive_cards.append(self.state_space[f"player_{i+1}_card_{j+1}"])
                else:
                    alive_cards.append("dead")
            
        print(f"Player 1: {alive_cards[0]} {alive_cards[1]} {self.state_space['player_1_coins']}")
        print(f"Player 2: {alive_cards[2]} {alive_cards[3]} {self.state_space['player_2_coins']}")
        print("----------------")



    def observe(self, agent):
        if agent == "player_1":
            return np.array(
                [self.state_space["player_1_card_1"],
                 self.state_space["player_1_card_2"],
                 self.state_space["player_1_coins"],
                 self.state_space["player_2_card_1_alive"],
                 self.state_space["player_2_card_2_alive"],
                 self.state_space["player_2_coins"],
                 self.state_space["player_1_action"],
                 self.state_space["player_2_action"], 
                 ]
                )
        elif agent == "player_2":
            return np.array(
                [self.state_space["player_2_card_1"],
                 self.state_space["player_2_card_2"],
                 self.state_space["player_2_coins"],
                 self.state_space["player_1_card_1_alive"],
                 self.state_space["player_1_card_2_alive"],
                 self.state_space["player_1_coins"],
                 self.state_space["player_2_action"],
                 self.state_space["player_1_action"], 
                 ]
                )
    
    def reset(self, seed=None, options=None):

        self.agents = [f"player_{i+1}" for i in range(2)]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        # allows stepping through the agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()



        
        # custom instructions
        self.deck = Deck(CARDS)
        self.state_space = {
            "player_1_card_1": self.deck.draw_card(),
            "player_1_card_2": self.deck.draw_card(),
            "player_2_card_1": self.deck.draw_card(),
            "player_2_card_2": self.deck.draw_card(),
            "player_1_card_1_alive": True,
            "player_1_card_2_alive": True,
            "player_2_card_1_alive": True,
            "player_2_card_2_alive": True,
            "player_1_coins": 1,
            "player_2_coins": 1,
            "player_1_action": None,
            "player_2_action": None
        }

        self.player_turn = 0


    def loose_card(self, agent, state_space:dict) -> None:
        """Loose a card for a player"""

        if(self.state_space[f"{agent}_card_1_alive"]):
            self.state_space[f"{agent}_card_1_alive"] = False
        else:
            self.state_space[f"{agent}_card_2_alive"] = False

            
    def process_action(self, agent, other_agent, action):
        action = self.get_action_string(action)

        if action == "income":
            self.state_space[f"{agent}_coins"] += 1
        elif action == "foreign_aid":
            self.state_space[f"{agent}_coins"] += 2
        elif action == "tax":
            self.state_space[f"{agent}_coins"] += 3
        elif action == "assassinate" and self.state_space[f"{agent}_coins"] >= 3:
            self.loose_card(other_agent, self.state_space)
            self.state_space[f"{agent}_coins"] -= 3
        elif action == "exchange":
            self.deck.add_card(self.state_space[f"{agent}_card_1"])
            self.deck.add_card(self.state_space[f"{agent}_card_2"])
            self.state_space[f"{agent}_card_1"] = self.deck.draw_card()
            self.state_space[f"{agent}_card_2"] = self.deck.draw_card()
        elif action == "steal":
            self.state_space[f"{agent}_coins"] += min(2, self.state_space[f"{other_agent}_coins"])
            self.state_space[f"{other_agent}_coins"] -= min(2, self.state_space[f"{other_agent}_coins"])
        elif action == "coup" and self.state_space[f"{agent}_coins"] >= 7:
            self.state_space[f"{agent}_coins"] -= 7
            self.loose_card(other_agent, self.state_space)
        else:
            print("invalid action")

    def set_game_result(self):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            reward =  (int(self.state_space[f"{name}_card_1_alive"]) + int(self.state_space[f"{name}_card_2_alive"])) - (int(self.state_space[f"{self.agents[1 - i]}_card_1_alive"]) + int(self.state_space[f"{self.agents[1 - i]}_card_2_alive"]))
            self.rewards[name] = reward


    def step(self, action):
        """Step requires checking if the other player counteracts or challenges before executing it"""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection

        self.player_turn = self.agent_name_mapping[agent]

        self._cumulative_rewards[agent] = 0


        self.state[self.agent_selection] = action

        
        #action_name = self.get_action_string(action)


        # give a new set of observations if the agent is the last one

        self.num_moves += 1

        self.truncations = {
            agent: self.num_moves >= NUM_ITERS for agent in self.agents
        }


            #self.observations[0] = s

            # for i in self.agents:


            #     self.observations[i] = self.state[
            #         self.agents[1 - self.agent_name_mapping[i]]
            #     ]
            # print(self.observations)
        # else:
        #     # necessary so that observe() returns a reasonable observation at all times.
        #     self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
        #     # no rewards are allocated until both players give an action
        #     self._clear_rewards()
            
        # change the agent selection to the next agent
        self.agent_selection = self._agent_selector.next()

        # process the action of the current agent
        self.process_action(agent, self.agent_selection, action)

        terminate = not((
            self.state_space["player_1_card_1_alive"] 
            or self.state_space["player_1_card_2_alive"])
            and (
            self.state_space["player_2_card_1_alive"] 
            or self.state_space["player_2_card_2_alive"]))



        if terminate:
            self.set_game_result()



        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


        # if (
        #     self.terminations[agent]
        #     or self.truncations[agent]
        # ):
        #     # handles stepping an agent which is already dead
        #     # accepts a None action for the one agent, and moves the agent_selection to
        #     # the next dead agent,  or if there are no more dead agents, to the next live agent
        #     #self._was_dead_step(action)

        #     # return reward which is the difference in number of cards alive
            
        #     if agent == "player_1":
        #         self.rewards["player_1"] = (
        #         (int(self.state_space["player_1_card_1_alive"]) + int(self.state_space["player_1_card_2_alive"]))
        #         - (int(self.state_space["player_2_card_1_alive"]) + int(self.state_space["player_2_card_2_alive"]))            
        #         )
        #     else:
        #         self.rewards["player_2"] = (
        #         (int(self.state_space["player_2_card_1_alive"]) + int(self.state_space["player_2_card_2_alive"]))
        #         - (int(self.state_space["player_1_card_1_alive"]) + int(self.state_space["player_1_card_2_alive"]))            
        #         )
                
            #self._was_dead_step(action)








        if self.render_mode == "human":
            self.render()