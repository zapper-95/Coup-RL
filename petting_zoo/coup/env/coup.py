from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
#from player import Player
from deck import Deck
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, MultiBinary
import gymnasium
import numpy as np


# based on code from https://pettingzoo.farama.org/content/environment_creation/


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
    "none"
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
            "player_2_coins": 2,
            "player_1_action": ACTIONS.index("none"),
            "player_2_action": ACTIONS.index("none")
        }

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


        self._action_spaces = {agent: Discrete(11) for agent in self.agents}

        # the players can see all of the state, but other players hands
        self._observation_spaces = {
            agent: Dict(
                {
                    "observation": MultiDiscrete([
                        len(CARDS),
                        len(CARDS),
                        2,
                        2,
                        13,
                        len(ACTIONS),
                        len(CARDS)+1,
                        len(CARDS)+1,
                        13,
                        len(ACTIONS)
                    ]),
                    "action_mask": MultiBinary(len(ACTIONS)),
                }
            )
            for agent in self.agents
        }


        
    def get_action_string(self, action_id:int) -> str:
        """Returns the string representation of an action."""
        if action_id is None:
            return None
        
        return ACTIONS[action_id]

    def update_player_action(self, action:dict[str, int]) -> None:
        """Updates the last action of a given player."""
        if "player_1" in action:
            self.state_space["player_1_action"] = action["player_1"]
        else:
            self.state_space["player_2_action"] = action["player_2"]

    
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]


    def render(self, action):
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
        print(f"Action: {self.get_action_string(action)}")
        # if self.player_turn == 0:
        #     print(f"Action: {self.state_space['player_1_action']}")
        # else:
        #     print(f"Action: {self.state_space['player_2_action']}")
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
        
        other_action = self.state_space[f"{other_agent}_action"]
        other_action_str = self.get_action_string(other_action)

        player_past_action = self.state_space[f"{agent}_action"]
        player_past_action_str = self.get_action_string(player_past_action)

        # get legal moves
        if other_action_str == "counteract":
            # can only challenge or pass
            legal_moves = [8, 9]
        elif other_action_str == "challenge" and player_past_action_str != "counteract":
            # if the other player challenged a normal action, pass a turn
            legal_moves = [9]   
        elif other_action_str in ["none", "pass"] or (other_action_str == "challenge" and player_past_action_str == "counteract"):
            # can do anything except counteract, challenge, or pass
            legal_moves = [0, 1, 2, 3, 4, 5, 6]
        else:
            # all moves except pass
            legal_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            

        if self.state_space[f"{agent}_coins"] >= 10:
            # have to coup, if legal
            if 6 in legal_moves:
                legal_moves = [6]



        if self.state_space[f"{agent}_coins"] < 7:
            # can't coup
            legal_moves = [x for x in legal_moves if x != 6]

        if self.state_space[f"{agent}_coins"] < 3:
            # can't assassinate
            legal_moves = [x for x in legal_moves if x != 3]

        if not self.can_counteract(other_action):
            # can't counteract
            legal_moves = [x for x in legal_moves if x != 7]

        if not self.can_challenge(other_action):
            # can't challenge
            legal_moves = [x for x in legal_moves if x != 8]


        action_mask = np.zeros(len(ACTIONS), "int8")

        for move in legal_moves:
            action_mask[move] = 1


        observation = np.array(
            [   CARDS.index(self.state_space[f"{agent}_card_1"]),
                CARDS.index(self.state_space[f"{agent}_card_2"]),
                int(self.state_space[f"{agent}_card_1_alive"]),
                int(self.state_space[f"{agent}_card_2_alive"]),
                self.state_space[f"{agent}_coins"],
                self.state_space[f"{agent}_action"],
                CARDS.index(self.state_space[f"{other_agent}_card_1"]),
                CARDS.index(self.state_space[f"{other_agent}_card_2"]),
                self.state_space[f"{other_agent}_coins"],
                self.state_space[f"{other_agent}_action"]
            ]
        )


        # # hide the other player's cards if they are alive
        if self.state_space[f"{other_agent}_card_1_alive"]:
            observation[6] = len(CARDS)
        
        if self.state_space[f"{other_agent}_card_2_alive"]:
            observation[7] = len(CARDS)



        return {"observation": observation, "action_mask": action_mask}
        

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
            "player_2_coins": 2,
            "player_1_action": ACTIONS.index("none"),
            "player_2_action": ACTIONS.index("none")
        }

        self.player_turn = 0


    def loose_card(self, agent:str) -> None:
        """Loose a card for a player"""

        if(self.state_space[f"{agent}_card_1_alive"]):
            self.state_space[f"{agent}_card_1_alive"] = False
        else:
            self.state_space[f"{agent}_card_2_alive"] = False
            
    def process_action(self, agent:str, other_agent:str, action:int) -> None:
        """Processes the action of a player, and updates the state space accordingly."""
        action_str = self.get_action_string(action)

        if action_str == "income":
            self.state_space[f"{agent}_coins"] += 1
        elif action_str == "foreign_aid":
            self.state_space[f"{agent}_coins"] += 2
        elif action_str == "tax":
            self.state_space[f"{agent}_coins"] += 3
        elif action_str == "assassinate" and self.state_space[f"{agent}_coins"] >= 3:
            self.loose_card(other_agent)
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
            self.loose_card(other_agent)
        elif action_str == "counteract" and self.can_counteract(self.state_space[f"{other_agent}_action"]):
            self.reverse_action(other_agent, agent, self.state_space[f"{other_agent}_action"])
            
        elif action_str == "challenge" and self.can_challenge(self.state_space[f"{other_agent}_action"]):
            if self.get_action_string(self.state_space[f"{other_agent}_action"]) != "counteract":
                # if a normal action is being challenged
                
                # check whether that action was legal
                    if self.action_legal(other_agent, self.state_space[f"{other_agent}_action"]):
                        self.loose_card(agent)
                    else:
                        # if the action was not legal, than reverse its effect
                        self.reverse_action(other_agent, agent, self.state_space[f"{other_agent}_action"])
                        self.loose_card(other_agent)
            else:
                # if it is a counteraction being challenged, check whether the counteraction was legal
                if self.counteraction_legal(self.state_space[f"{agent}_action"], other_agent):
                    self.loose_card(agent)
                else:
                    # if it was not legal, let the player play their initial action again
                    self.process_action(agent, other_agent, self.state_space[f"{agent}_action"])
                    self.loose_card(other_agent)
        elif action_str == "pass":
            pass        
        else:
            # action did not go through
            action = ACTIONS.index("none")



        self.update_player_action({agent:action})

                

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
            if not self.state_space[f"{agent}_card_2_alive"]:
                self.state_space[f"{agent}_card_2_alive"] = True
            elif not self.state_space[f"{agent}_card_1_alive"]:
                self.state_space[f"{agent}_card_1_alive"] = True
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

    def set_game_result(self, agent:int) -> None:
        """Sets termination to true, and calculates the rewards for a given player."""
        self.terminations[agent] = True
        other_indx = 1-self.agents.index(agent)

        reward = ((
                int(self.state_space[f"{agent}_card_1_alive"]) 
                + int(self.state_space[f"{agent}_card_2_alive"])
                )- 
                (
                int(self.state_space[f"{self.agents[other_indx]}_card_1_alive"]) 
                + int(self.state_space[f"{self.agents[other_indx]}_card_2_alive"])
                ))
        self.rewards[agent] = reward
    
    def reset_game_result(self):
        """If the other player moves out of a terminated state, reset the termination and rewards."""
        for agent in self.agents:
            self.terminations[agent] = False
            self.rewards[agent] = 0


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
        # only keep the cards that are alive
        cards = [card for card in cards if self.state_space[f"{agent}_card_{cards.index(card)+1}_alive"]]

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
        
        # reset the cumulative rewards for both agents, as we recalculate them at the end
        for a in self.agents:
            self._cumulative_rewards[a] = 0

        self.num_moves += 1

        self.truncations = {
            agent: self.num_moves >= NUM_ITERS for agent in self.agents
        }

        # get the next agent
        self.agent_selection = self._agent_selector.next()


        


        # stops the action if it is not counteract or challenged if the player is currently in a dead state
        take_action = (
                        (not self.terminated()
                        or (self.terminated()
                        and (self.get_action_string(action)
                        in self.final_actions))))


        if take_action:
            self.process_action(agent, self.agent_selection, action)        

        if self.terminated():
            self.set_game_result(agent)
        else:
            self.reset_game_result()      
        
        if take_action and self.get_action_string(action) != "pass" and self.render_mode == "human":
            self.render(action)

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()