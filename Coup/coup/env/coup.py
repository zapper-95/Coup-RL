from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, MultiBinary
import gymnasium
import numpy as np
import random


# pettingzoo environment setup adapted from this code https://pettingzoo.farama.org/content/environment_creation/


CARDS = ["ambassador", "assassin", "captain", "contessa", "duke"]
ACTIONS = [
    "income",
    "foreign_aid",
    "tax",
    "assassinate",
    "exchange",
    "steal",
    "coup",
    "counteract",
    "challenge",
    "pass",
    "none",
    "kill_card_1",
    "kill_card_2",
]
NUM_ITERS = 100

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
        "name": "coup_v2",
    }

    def __init__(self, render_mode=None, k_actions=4):
        assert k_actions >= 4
        self.k_actions = k_actions

        self.render_mode = render_mode
    

        self.action_card = {
            "tax": "duke",
            "assassinate": "assassin",
            "exchange": "ambassador",
            "steal": "captain",
        }


        self.action_counter_card = {
            "foreign_aid":["duke"],
            "assassinate":["contessa"],
            "steal": ["ambassador", "captain"],   
        }

        self.final_actions = ["counteract", "challenge"]

        self.no_challenge_actions = ["income", "foreign_aid", "coup", "none", "challenge"]

        self.player_turn = 0
        self.agents = [f"player_{i+1}" for i in range(2)]
        self.possible_agents = self.agents[:]


        # dictionary of the agent names to their index
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.agents))))
        )
        self.deck = Deck(CARDS)


        self.action_spaces = {agent: Discrete(len(ACTIONS)) for agent in self.agents}

        # the players can see all of the state, but other players hands
        self.observation_spaces = {
            agent: Dict(
                {
                    "observations": MultiDiscrete([
                        len(CARDS),
                        len(CARDS),
                        2,
                        2,
                        13,
                        len(CARDS)+1,
                        len(CARDS)+1,
                        13,
                        *[len(ACTIONS) for _ in range(self.k_actions)]
                    ]),
                    "action_mask": MultiBinary(len(ACTIONS)),
                }
            )
            for agent in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        
    def get_action_string(self, action_id:int) -> str:
        """Returns the string representation of an action."""
        if action_id is None:
            return None
        
        return ACTIONS[action_id]
    
    def get_action_id(self, action:str) -> int:
        """Returns the id of an action."""
        if action in ACTIONS:
            return ACTIONS.index(action)
        return None
    def get_card(self, card_id:int) -> str:
        """Returns the string representation of a card."""
        if card_id is None or card_id >= len(CARDS):
            return None
        
        return CARDS[card_id]


    def update_action_history(self, action:int) -> None:
        """Updates the action history of the environment."""
        self.action_history.append(action)

        if len(self.action_history)> self.k_actions:
            self.action_history.pop(0)


    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]


    def render(self, action=10, display_action=True):
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
        if display_action:
            print(f"Action: {self.get_action_string(action)}")
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
        print()
        print()

    


    def observe(self, agent):
        """Returns the observation of a given player. This is imperfect information, as the player cannot see the other player's cards."""
        other_agent = "player_2" if agent == "player_1" else "player_1"

        legal_moves = []
        
        other_past_action = self.action_history[-3]
        player_past_action = self.action_history[-2]
        other_action = self.action_history[-1]

        other_past_action_str = self.get_action_string(other_past_action)
        player_past_action_str = self.get_action_string(player_past_action)
        other_action_str = self.get_action_string(other_action) 

        normal_actions = [
                self.get_action_id("income"),
                self.get_action_id("foreign_aid"),
                self.get_action_id("tax"),
                self.get_action_id("assassinate"),
                self.get_action_id("exchange"),
                self.get_action_id("steal"),
                self.get_action_id("coup"),
            ]


        # cases where you take a normal action
        if ([other_past_action_str, player_past_action_str, other_action_str] in 
            [
                ["none", "none", "none"],
                *[ [self.get_action_string(action), "pass", "pass"] for action in normal_actions ],

                *[ [self.get_action_string(action), "challenge", kill] for action in normal_actions for kill in ["kill_card_1", "kill_card_2"] ],

                *[ ["pass", kill, "pass"] for kill in ["kill_card_1", "kill_card_2"] ],

                *[ [self.get_action_string(action), "counteract", "pass"] for action in normal_actions ],

                *[ ["challenge", kill, "pass"] for kill in ["kill_card_1", "kill_card_2"] ],

                *[ [self.get_action_string(action), kill, "pass"] for action in normal_actions for kill in ["kill_card_1", "kill_card_2"] ],
            ]

            or 
                [self.get_action_string(self.action_history[-4]), other_past_action_str, player_past_action_str, other_action_str] in
                [["counteract", "challenge", "pass", kill] for kill in ["kill_card_1", "kill_card_2"]]
               
        ):
            legal_moves = normal_actions

            if self.state_space[f"{agent}_coins"] >= 10:
                legal_moves = [self.get_action_id("coup")]
            elif self.state_space[f"{agent}_coins"] < 7:
                legal_moves = [x for x in legal_moves if x != self.get_action_id("coup")]
            
            if self.state_space[f"{agent}_coins"] < 3:
                legal_moves = [x for x in legal_moves if x != self.get_action_id("assassinate")]


        # cases where the other player played a normal action
        elif other_action_str in [self.get_action_string(x) for x in normal_actions]:

            if other_action_str == "assassinate" or other_action_str == "coup":
                if self.state_space[f"{agent}_card_1_alive"]:
                    legal_moves.append(self.get_action_id("kill_card_1"))
                
                if self.state_space[f"{agent}_card_2_alive"]:
                    legal_moves.append(self.get_action_id("kill_card_2"))
            else:
                legal_moves = [self.get_action_id("pass")]

            if self.can_challenge(other_action):
                legal_moves.append(self.get_action_id("challenge"))

            if self.can_counteract(other_action):
                legal_moves.append(self.get_action_id("counteract"))




        # cases where you can only pass
        elif (
                
                # Other player passed counteracting or challenging your action
                player_past_action_str in [self.get_action_string(x) for x in normal_actions] and other_action_str == "pass"
                

                # you correctly challenged a counteraction, and so you must pass your turn so the other player can take theirs
                or [other_past_action_str, player_past_action_str, other_action_str] in 
                [["counteract", "challenge", "kill_card_1"],
                ["counteract", "challenge", "kill_card_2"]
                ]


                # you played an action that killed the other player's card, so you must pass so it is their turn
                or [player_past_action_str, other_action_str] in 
                [
                    ["assassinate", "kill_card_1"],
                    ["assassinate", "kill_card_2"],
                    ["coup", "kill_card_1"],
                    ["coup", "kill_card_2"]
                ]


                or [self.get_action_string(self.action_history[-4]), other_past_action_str, player_past_action_str, other_action_str] in
                [[self.get_action_string(action), "challenge", "pass", kill] for action in normal_actions for kill in ["kill_card_1", "kill_card_2"]]
              ):
            legal_moves = [self.get_action_id("pass")]



        # where the other player played a counteraction
        elif other_action_str == "counteract":
            # can only challenge or pass
            legal_moves = [self.get_action_id("challenge"), self.get_action_id("pass")]
        

        # cases where you or the other player will loose a card
        elif other_action_str == "coup" or other_action_str == "challenge" or [player_past_action_str, other_action_str] == ["challenge", "pass"]:

            # if true, you lost the challenge
            if self.state_space[f"{agent}_loose_card"] > 0:

                if self.state_space[f"{agent}_card_1_alive"]:
                    legal_moves.append(self.get_action_id("kill_card_1"))
                
                if self.state_space[f"{agent}_card_2_alive"]:
                    legal_moves.append(self.get_action_id("kill_card_2"))
            else:
                # if you won the challenge, pass so the other player looses a card
                legal_moves = [self.get_action_id("pass")]
            


        action_mask = np.zeros(len(ACTIONS), "int8")

        for move in legal_moves:
            action_mask[move] = 1


        observation = np.array(
            [   CARDS.index(self.state_space[f"{agent}_card_1"]),
                CARDS.index(self.state_space[f"{agent}_card_2"]),
                int(self.state_space[f"{agent}_card_1_alive"]),
                int(self.state_space[f"{agent}_card_2_alive"]),
                self.state_space[f"{agent}_coins"],
                CARDS.index(self.state_space[f"{other_agent}_card_1"]) if not self.state_space[f"{other_agent}_card_1_alive"] else len(CARDS),
                CARDS.index(self.state_space[f"{other_agent}_card_2"]) if not self.state_space[f"{other_agent}_card_2_alive"] else len(CARDS),
                self.state_space[f"{other_agent}_coins"],

            ]
        )


        # array with actions padded on with nones, up to k actions
        action_history_array = np.array([ACTIONS.index("none") for _ in range(self.k_actions)])
        action_history_array[-len(self.action_history):] = self.action_history
        
        observation = np.concatenate((observation, action_history_array))

        return {"observations": observation, "action_mask": action_mask}
        

    def reset(self, seed=None, options=None):

        self.agents = [f"player_{i+1}" for i in range(2)]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.num_moves = 0

        # allows stepping through the agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.prev_winner = None
        self.prev_reward = 0


        
        self.deck = Deck(CARDS, seed)
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
            "player_2_coins": 2,
            "player_1_loose_card": 0,
            "player_2_loose_card": 0,
        }

        self.action_history = [ACTIONS.index("none") for _ in range(self.k_actions)]

        self.player_turn = 0


    def loose_card(self, agent:str, card:int) -> None:
        """Player looses a card. If the card is already dead, the other card is killed instead."""
        

        if self.state_space[f"{agent}_loose_card"] == 1:
            if self.state_space[f"{agent}_card_{card}_alive"]:
                self.state_space[f"{agent}_card_{card}_alive"] = False
            else:
                print("Error: Card already dead. Killing other")
                self.state_space[f"{agent}_card_{3-card}_alive"] = False
        else:
            # in the case the player is set to loose both their cards
            self.state_space[f"{agent}_card_1_alive"] = False
            self.state_space[f"{agent}_card_2_alive"] = False  

        
        self.state_space[f"{agent}_loose_card"] = 0



    def set_loose_card(self, agent:str) -> None:
        """Set the state, so that a player is set to loose this many cards"""
        self.state_space[f"{agent}_loose_card"] += 1

            
    def process_action(self, agent:str, other_agent:str, action:int, update_history=True) -> None:
        """Processes the action of a player, and updates the state space accordingly."""
        action_str = self.get_action_string(action)
        agent_past_action = self.action_history[-2]
        other_player_action = self.action_history[-1]


        if action_str == "income":
            self.state_space[f"{agent}_coins"] += 1
        elif action_str == "foreign_aid":
            self.state_space[f"{agent}_coins"] += 2
        elif action_str == "tax":
            self.state_space[f"{agent}_coins"] += 3
        elif action_str == "assassinate" and self.state_space[f"{agent}_coins"] >= 3:
            self.set_loose_card(other_agent)
            self.state_space[f"{agent}_coins"] -= 3
        elif action_str == "exchange":

            if self.state_space[f"{agent}_card_1_alive"]:
                self.deck.add_card(self.state_space[f"{agent}_card_1"])
                self.state_space[f"{agent}_card_1"] = self.deck.draw_card()
            
            if self.state_space[f"{agent}_card_2_alive"]:
                self.deck.add_card(self.state_space[f"{agent}_card_2"])
                self.state_space[f"{agent}_card_2"] = self.deck.draw_card()

        elif action_str == "steal":
            self.state_space[f"{agent}_coins"] += min(2, self.state_space[f"{other_agent}_coins"])
            self.state_space[f"{other_agent}_coins"] -= min(2, self.state_space[f"{other_agent}_coins"])
        elif action_str == "coup" and self.state_space[f"{agent}_coins"] >= 7:
            self.state_space[f"{agent}_coins"] -= 7
            self.set_loose_card(other_agent)
        elif action_str == "counteract" and self.can_counteract(other_player_action):
            self.reverse_action(other_agent, agent, other_player_action)
            
        elif action_str == "challenge" and self.can_challenge(other_player_action):
            if self.get_action_string(other_player_action) != "counteract":
                # if a normal action is being challenged
                
                # check whether that action was legal
                    if self.action_legal(other_agent, other_player_action):
                        self.set_loose_card(agent)
                    else:
                        # if the action was not legal, than reverse its effect
                        self.reverse_action(other_agent, agent, other_player_action)
                        self.set_loose_card(other_agent)
            else:
                # if it is a counteraction being challenged, check whether the counteraction was legal
                if self.counteraction_legal(agent_past_action, other_agent):
                    self.set_loose_card(agent)
                else:
                    # if it was not legal, let the player play their initial action again (without having to pay again if it was assissinate)
                    if self.get_action_string(agent_past_action) == "assassinate":
                        self.state_space[f"{agent}_coins"] += 3

                    self.process_action(agent, other_agent, agent_past_action, update_history=False)
                    self.set_loose_card(other_agent)
        elif action_str == "pass":
            pass
        elif action_str == "kill_card_1":
            self.loose_card(agent, 1)
        elif action_str == "kill_card_2":
            self.loose_card(agent, 2)     
        else:
            # action did not go through
            action = ACTIONS.index("none")


        if update_history:
            self.update_action_history(action)

                

    def reverse_action(self, other_agent:str, agent:str, action:int) -> None:
        """Reverse the action of the other player."""
        action = self.get_action_string(action)




        if action == "steal":
            self.state_space[f"{agent}_coins"] += min(2, self.state_space[f"{other_agent}_coins"])
            self.state_space[f"{other_agent}_coins"] -= min(2, self.state_space[f"{other_agent}_coins"])
        elif action == "tax":
            self.state_space[f"{other_agent}_coins"] -= 3
        elif action == "foreign_aid":
            self.state_space[f"{other_agent}_coins"] -= 2
        elif action == "assassinate":
            self.state_space[f"{agent}_loose_card"] = 0
        elif action == "exchange":
            card_1, card_2 = self.deck.draw_bottom_card(), self.deck.draw_bottom_card()
            
            self.deck.add_card(self.state_space[f"{other_agent}_card_1"])
            self.deck.add_card(self.state_space[f"{other_agent}_card_2"])
            
            self.state_space[f"{other_agent}_card_1"] = card_1
            self.state_space[f"{other_agent}_card_2"] = card_2





    def can_challenge(self, action:int) -> bool:
        """Check if an action can be challenged."""
        if self.get_action_string(action) in self.no_challenge_actions:
            return False
        return True

    def set_game_result(self) -> None:
        """Sets termination to true."""

        for agent in self.agents:
            self.terminations[agent] = True

    def get_reward(self, agent:str, other_agent:str)->int:
        return int(
                    int(self.state_space[f"{agent}_card_1_alive"]) 
                    + int(self.state_space[f"{agent}_card_2_alive"])
                    )- (
                    int(self.state_space[f"{other_agent}_card_1_alive"]) 
                    + int(self.state_space[f"{other_agent}_card_2_alive"])
                    )

    def get_current_winner(self) -> str:
        if self.state_space["player_1_card_2_alive"]:
            return "player_1"
        else:
            return "player_2"
    
    def action_legal(self, agent:int, action:int) -> bool:
        """Check if an action of given player is legal for the cards they have."""
        action = self.get_action_string(action)
        alive_cards = []

        if action != "exchange":
            for j in range(2):
                if self.state_space[f"{agent}_card_{j+1}_alive"]:
                    alive_cards.append(self.state_space[f"{agent}_card_{j+1}"])
        else:
            temp = [self.deck.draw_bottom_card(), self.deck.draw_bottom_card()]
            
            if not self.state_space[f"{agent}_card_1_alive"]:
                self.deck.add_card(temp.pop())
            
            # store a copy and return both cards to the deck
                
            alive_cards = temp.copy()

            for card in temp:
                self.deck.add_card(card)
            

        if action in self.action_card.keys():
            if not self.action_card[action] in alive_cards:
                return False
                
            
        return True
        
    def can_counteract(self, action:int) -> bool:
        """Check if an action can be counteracted"""
        action = self.get_action_string(action)
        return action in self.action_counter_card.keys()

    
    def terminated(self) -> None:
        """Check if the game is in a termination state."""
        return not((
        self.state_space["player_1_card_1_alive"] 
        or self.state_space["player_1_card_2_alive"])
        and (
        self.state_space["player_2_card_1_alive"] 
        or self.state_space["player_2_card_2_alive"]))

    def counteraction_legal(self, stop_action:int, agent:int) -> bool:
        """Check if a player legally counteracted the action of another"""
        
        # cards of the counteracting player
        cards = [self.state_space[f"{agent}_card_1"], self.state_space[f"{agent}_card_2"]]

        # only keep the first card if it is alive
        if not self.state_space[f"{agent}_card_1_alive"]:
            cards.pop(0)
        # check that the action they are stopping can be counteracted
        if self.can_counteract(stop_action):
            # action which is being stopped
            stop_action = self.get_action_string(stop_action)

            # check that they have at least one of the required cards to stop the action

            for card in cards:
                if card in self.action_counter_card[stop_action]:
                    return True
        return False
    


    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection

        self.player_turn = self.agent_name_mapping[agent]

        for a in self.agents:
            self.rewards[a] = 0

        self.num_moves += 1

        self.truncations = {
            agent: self.num_moves >= NUM_ITERS for agent in self.agents
        }

        # get the next agent
        self.agent_selection = self._agent_selector.next()
        other_agent = self.agent_selection


        self.process_action(agent, other_agent, action)        


     
        if self.render_mode == "human":
            self.render(action)


        reward = self.get_reward(agent, other_agent)
        if self.terminated():
            self.rewards[agent], self.rewards[other_agent] = reward, -reward
            self.set_game_result()

        self._accumulate_rewards()

class Deck():
    def __init__(self, cards, seed=None) -> None:
        random.seed(seed)

        deck = [element for element in cards for _ in range(3)]
        self.deck  = deck
        self.shuffle()

    def draw_card(self):
        return self.deck.pop(0)
    
    def add_card(self, card):
        self.deck.append(card)

    def draw_bottom_card(self):
        """For when exchange is correctly challenged"""
        return self.deck.pop()
    
    def peek_card(self, index):
        return self.deck[index]

    def shuffle(self):
        random.shuffle(self.deck)