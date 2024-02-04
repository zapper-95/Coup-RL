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
NUM_ITERS = 10
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
        
        self.player_1_card_1 = None
        self.player_1_card_2 = None
        self.player_2_card_1 = None
        self.player_2_card_2 = None
        self.player_1_card_1_alive = True
        self.player_1_card_2_alive = True
        self.player_2_card_1_alive = True
        self.player_2_card_2_alive = True
        self.player_1_coins = 1
        self.player_2_coins = 1
        self.player_1_proposed_action = None
        self.player_2_proposed_action = None

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
        return ACTIONS[action_id]

    def update_state(self, actions):
        """Updates the state space of the environment"""
        if "player_1" in actions:
            self.player_1_proposed_action = actions["player_1"]
        else:
            self.player_2_proposed_action = actions["player_2"]

    def remove_proposed_actions(self):
        self.player_1_proposed_action = None
        self.player_2_proposed_action = None
    
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


        alive_cards = [self.player_1_card_1, self.player_1_card_2, self.player_2_card_1, self.player_2_card_2]
        print("----------------")
        print(f"Player 1: {alive_cards[0]} {alive_cards[1]} {self.player_1_coins}")
        print(f"Player 2: {alive_cards[2]} {alive_cards[3]} {self.player_1_coins}")
        print("----------------")
        if self.player_turn == 0:
            print(f"Action: {self.player_1_proposed_action}")
        else:
            print(f"Action: {self.player_2_proposed_action}")
        print("----------------")
        print()


    def observe(self, agent):
        if agent == "player_1":
            return np.array(
                [self.player_1_card_1,
                 self.player_1_card_2,
                 self.player_1_coins,
                 self.player_2_card_1_alive,
                 self.player_2_card_2_alive,
                self.player_2_coins,
                self.player_1_proposed_action,
                self.player_2_proposed_action, 
                 ]
                )
        elif agent == "player_2":
            return np.array(
                [self.player_2_card_1,
                 self.player_2_card_2,
                 self.player_2_coins,
                 self.player_1_card_1_alive,
                 self.player_1_card_2_alive,
                 self.player_1_coins,
                 self.player_2_proposed_action,
                 self.player_1_proposed_action, 
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
        self.player_1_card_1 = self.deck.draw_card()
        self.player_1_card_2 = self.deck.draw_card()
        self.player_2_card_1 = self.deck.draw_card()
        self.player_2_card_2 = self.deck.draw_card()

        self.player_1_card_1_alive = True
        self.player_1_card_2_alive = True
        self.player_2_card_1_alive = True
        self.player_2_card_2_alive = True

        self.player_1_coins = 1
        self.player_2_coins = 1

        self.player_1_proposed_action = None
        self.player_2_proposed_action = None

        self.player_turn = 0


    def step(self, action):
        """Step requires checking if the other player counteracts or challenges before executing it"""
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return     

        agent = self.agent_selection

        self.player_turn = self.agent_name_mapping[agent]

        self._cumulative_rewards[agent] = 0


        self.state[self.agent_selection] = action
        

        
        #action_name = self.get_action_string(action)


        # give a new set of observations if the agent is the last one
        if True:
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


        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()