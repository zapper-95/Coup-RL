import random
import gym
from gym import Env, spaces

from deck import Deck
from player import Player

class CoupEnv(Env):
    """
    States:
    
    State is a tuple of
    1. Player 1's first card name
    2. Player 2's first card name
    2. Player 1's second card name
    3. Player 2's second card name
    4. State of player 1's first card (dead or alive)
    5. State of player 1's second card (dead or alive)
    6. Player 1's coins
    7. Player 2's coins
    8. Which player's turn it is
    """

    """
    Actions:
    1. Income
    2. Foreign Aid
    3. Coup
    4. Tax
    5. Assassinate
    6. Exchange
    7. Steal
    8. Block Assissination
    9. Block Stealing
    10. Block Foreign Aid
    """
    def __init__(self):
        super(CoupEnv, self).__init__()

        self.observation_space = spaces.Dict({
            "player_1_card_1_name": spaces.Discrete(5),
            "player_2_card_1_name": spaces.Discrete(5),
            "player_1_card_2_name": spaces.Discrete(5),
            "player_2_card_2_name": spaces.Discrete(5),
            "player_1_card_1_state": spaces.MultiBinary(2),
            "player_2_card_1_state": spaces.MultiBinary(2),
            "player_1_card_2_state": spaces.MultiBinary(2),
            "player_2_card_2_state": spaces.MultiBinary(2),
            "player_1_coins": spaces.Discrete(12),
            "player_2_coins": spaces.Discrete(12),
            "player_turn": spaces.MultiBinary(1)
        })

        self.action_space = spaces.Discrete(10)

    def _get_obs(self):
        return {
            "player_1_card_1_name": self.players[0].card_1.get_name(),
            "player_1_card_2_name": self.players[0].card_2.get_name(),

            "player_2_card_1_name": self.players[1].card_1.get_name(),
            "player_2_card_2_name": self.players[1].card_2.get_name(),

            "player_1_card_1_state": self.players[0].card_1.is_dead(),
            "player_1_card_2_state": self.players[0].card_2.is_dead(),

            
            "player_2_card_1_state": self.players[1].card_1.is_dead(),
            "player_2_card_2_state": self.players[1].card_2.is_dead(),

            "player_1_coins": self.players[0].get_coins(),
            "player_2_coins": self.players[1].get_coins(),

            "player_turn": self.player_turn
        }
    
    def reset(self):
        self.deck = Deck()

        self.players = [
                        Player("Player 1", self.deck), 
                        Player("Player 2", self.deck)
                        ]
        self.player_turn = 0

        observation = self._get_obs()
        return observation


    def _action_to_string(self, action):
        actions = {
            0: "Income",
            1: "Foreign Aid",
            2: "Coup",
            3: "Tax",
            4: "Assassinate",
            5: "Exchange",
            6: "Steal",
            7: "Block Assissination",
            8: "Block Stealing",
            9: "Block Foreign Aid"
        }
        return actions.get(action, "Invalid Action")


    def step(self, action):
        terminated = False
        reward = 0


        # play the current players action
        if self.players[self.player_turn].get_coins() >= 10:
            action = "Coup"
        else:
            action = self._action_to_string(action)

        actions = {
            "Income": self.players[self.player_turn].income,
            "Foreign Aid": self.players[self.player_turn].foreign_aid,
            "Tax": self.players[self.player_turn].tax,
            "Coup": lambda: self.players[self.player_turn].coup(self.players[1 - self.player_turn]),
            "Assassinate": lambda: self.players[self.player_turn].assasinate(self.players[1 - self.player_turn]),
            "Exchange": lambda: self.players[self.player_turn].exchange(self.deck),
            "Steal": lambda: self.players[self.player_turn].steal(self.players[1 - self.player_turn])
        }

        actions.get(action, lambda: None)()

        # switch turns
        self.player_turn = 1 - self.player_turn

        # check if game is over
        if self.players[0].card_1.is_dead() and self.players[0].card_2.is_dead():
            reward = -1
            terminated = True
        elif self.players[1].card_1.is_dead() and self.players[1].card_2.is_dead():
            reward = 1
            terminated = True

        observation = self._get_obs()

        return observation, reward, terminated, {}




def main():
    env = CoupEnv()
    obs = env.reset()
    
    terminated = False
    while not terminated:
        for _ in env.players:
            action = random.randint(0, 6)
            print(env._action_to_string(action))
            obs, reward, terminated, _ = env.step(action)
            
            for key in obs.keys():
                print(key, obs[key])
            print()

            if terminated:
                break

if __name__ == '__main__':
    main()