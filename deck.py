import random
class Deck():
    def __init__(self, cards) -> None:
        deck = [element for element in cards for _ in range(3)]
        random.shuffle(deck)
        self.deck  = deck

    def draw_card(self):
        return self.deck.pop()
    
    def add_card(self, card):
        self.deck.append(card)
        random.shuffle(self.deck)

