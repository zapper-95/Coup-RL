import random
from card import Card
from action import Action, Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal, Challenge, Counteract
class Player(object):

    
    def __init__(self, name, deck):
        self.name = name

        # 2 face down cards
        self._card_1 = deck.draw()
        self._card_2 = deck.draw()

        self._coins = 1
        self._last_action = None


    def choose_action(self) -> Action:
        # generate a random number between 0 and and 6
        decision = random.randint(0, 6)

        return self._action_to_object(decision)


    def choose_counteraction(self) -> Counteract:
        decision = random.randint(0, 1)

        return self._counteraction_to_object(decision)


    def choose_challenge(self) -> Challenge:
        decision = random.randint(0, 1)

        return self._challenge_to_object(decision)


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




    def _counteraction_to_object(self, action):
        counteractions = {
            0: None,
            1: Counteract(),
        }


    def _challenge_to_object(self, action):
        counteractions = {
            0: None,
            1: Challenge(),
        }


    def get_last_action(self) -> Action:
        return self._last_action

    def set_last_action(self, action):
        self._last_action = action    

    def get_coins(self) -> int:
        return self._coins
    
    def add_coins(self, amount:int) -> None:
        self._coins += amount

    def get_card_1(self) -> Card:
        return self._card_1
    
    def get_card_2(self) -> Card:
        return self._card_2
    
    def set_card_1(self, card:Card) -> None:
        self._card_1 = card
    
    def set_card_2(self, card:Card) -> None:
        self._card_2 = card

    #TODO: Change this to choose which card to lose. Currently removes card 1 then card 2
    def lose_card(self) -> None:
        if not self._card_1.is_dead():
            self._card_1.kill()
        else:
            self._card_2.kill()

