import random
class Deck():
    def __init__(self, cards) -> None:
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


