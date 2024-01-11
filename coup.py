import random
import gym
from gym import Env, spaces

from deck import Deck
from player import Player
from action import Action, Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal

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
            "player_1_card_1_dead": spaces.MultiBinary(1),
            "player_2_card_1_dead": spaces.MultiBinary(1),
            "player_1_card_2_dead": spaces.MultiBinary(1),
            "player_2_card_2_dead": spaces.MultiBinary(1),
            "player_1_coins": spaces.Discrete(12),
            "player_2_coins": spaces.Discrete(12),
            "player_turn": spaces.MultiBinary(1)
        })

        self.action_space = spaces.Discrete(10)

    def _get_obs(self):
        return {
            "player_1_card_1_name": self.players[0].get_card_1().get_name(),
            "player_1_card_2_name": self.players[0].get_card_2().get_name(),

            "player_2_card_1_name": self.players[1].get_card_1().get_name(),
            "player_2_card_2_name": self.players[1].get_card_2().get_name(),

            "player_1_card_1_dead": self.players[0].get_card_1().is_dead(),
            "player_1_card_2_dead": self.players[0].get_card_2().is_dead(),

            
            "player_2_card_1_dead": self.players[1].get_card_1().is_dead(),
            "player_2_card_2_dead": self.players[1].get_card_2().is_dead(),

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


    def _action_to_object(self, action):
        actions = {
            0: Income(),
            1: ForeignAid(),
            2: Coup(),
            3: Tax(),
            4: Assassinate(),
            5: Exchange(),
            6: Steal(),
        }
        return actions.get(action, "Invalid Action")


    def step(self, action):
        terminated = False
        reward = 0
        

        if type(action) in [Assassinate, Steal, Coup]:
            action.execute(self.players[self.player_turn], self.players[1 - self.player_turn])
        elif type(action) == Exchange:
            action.execute(self.players[self.player_turn], self.deck)
        else:
            action.execute(self.players[self.player_turn])



        self.player_turn = 1 - self.player_turn


        if self.players[0].get_card_1().is_dead() and self.players[0].get_card_2().is_dead():
            reward = -1
            terminated = True
        elif self.players[1].get_card_1().is_dead() and self.players[1].get_card_2().is_dead():
            reward = 1
            terminated = True

        observation = self._get_obs()

        return observation, reward, terminated, {}

    def challenge(self, action, player, other_player):
        if action in player.card_1.actions() or action in player.card_2.actions():
            other_player.lose_card()
        else:
            player.lose_card()

def main():
    env = CoupEnv()
    obs = env.reset()
    
    terminated = False
    while not terminated:
        for i, player in enumerate(env.players):

            action = env._action_to_object(random.randint(0, 6))

            print(action.get_name())
            obs, reward, terminated, _ = env.step(action)
            
            for key in obs.keys():
                print(key, obs[key])


            if terminated:
                break

if __name__ == '__main__':
    main()