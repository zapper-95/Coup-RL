import random
class Deck():
    def __init__(self, cards) -> None:
        deck = [element for element in cards for _ in range(3)]
        self.deck  = deck
        self.shuffle()

    def draw_card(self):
        random.shuffle(self.deck)
        return self.deck.pop()
    
    def add_card(self, card):
        self.deck.append(card)
        #random.shuffle(self.deck)

    def draw_card_no_shuffle(self):
        """For when exchange is correctly challenged"""
        return self.deck.pop()

    def shuffle(self):
        random.shuffle(self.deck)


