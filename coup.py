import random
import gym
from gym import Env, spaces

from deck import Deck
from player import Player
from action import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal, Challenge, Counteract
import copy
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
    9 . Player 1's last action
    10. Player 2's last action
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
    9. Counteract 
    11. Challenge
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
            "player_turn": spaces.MultiBinary(1),
            "player_1_last_action": spaces.Discrete(11),
            "player_2_last_action": spaces.Discrete(11)
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

            "player_turn": self.player_turn,
            "player_1_last_action": self.players[0].get_last_action(),
            "player_2_last_action": self.players[1].get_last_action()
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
            7: Challenge(),
            8: Counteract()
        }
        return actions.get(action, "Invalid Action")


    def step(self, action, turn):
        terminated = False
        reward = 0
        




        if type(action) in [Assassinate, Steal, Coup, Challenge]:
            action.execute(self.players[turn], self.players[1 - turn])
        elif type(action) == Exchange:
            action.execute(self.players[turn], self.deck)
        else:
            action.execute(self.players[turn])






        if self.players[0].get_card_1().is_dead() and self.players[0].get_card_2().is_dead():
            reward = -1
            terminated = True
        elif self.players[1].get_card_1().is_dead() and self.players[1].get_card_2().is_dead():
            reward = 1
            terminated = True

        observation = self._get_obs()
        
        return observation, reward, terminated, {}


def is_legal_action(action, action_queue):
    legal = True

    if type(action) == Challenge:
        legal =  action_queue != [] and action_queue[-1].can_be_challenged()  
    elif type(action) == Counteract:
        legal = action_queue != [] and action_queue[-1].can_be_counteracted() 
    
    return legal


def main():
    env = CoupEnv()
    obs = env.reset()
    
    terminated = False

    action_queue = []
    turn_queue = []

    while not terminated:
        for i, player in enumerate(env.players):

            print(i)
            action = env._action_to_object(random.randint(5, 8))
            while not is_legal_action(action, action_queue):
                action = env._action_to_object(random.randint(5,8))


            if action_queue != [] and i != turn_queue[-1]:  

                if (type(action_queue[-1]) == Challenge):
                    continue
                if type((action_queue[-1]) == Counteract):
                    continue



            print(action.get_name())
            

            
            if type(action) == Challenge:

                if action.good_challenge(env.players[1-i]):
                    action_queue = action_queue[0:-1]
                    turn_queue = turn_queue[0:-1]
                else:
                    if len(action_queue) == 1:
                        pass
                    elif len(action_queue) == 2:
                        action_queue = action_queue[1:]
                        turn_queue = turn_queue[1:]
            elif type(action) != Counteract:
                
                
                for act, turn in zip(action_queue, turn_queue):
                    obs, reward, terminated, _ = env.step(act, turn)
                action_queue = []

            action_queue.append(action)
            turn_queue.append(i)
            player.set_last_action("hello")

            if terminated:
                break

if __name__ == '__main__':
    main()