import random
from card import Card
class Player(object):

    
    def __init__(self, name, deck):
        self.name = name

        # 2 face down cards
        self._card_1 = deck.draw()
        self._card_2 = deck.draw()

        self._coins = 1

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

