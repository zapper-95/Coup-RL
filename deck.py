import random
from card import Card

class Deck(object):
    def __init__(self):
        self.cards = [
            Card("Ambassador"),
            Card("Ambassador"),
            Card("Ambassador"),

            Card("Assassin"),
            Card("Assassin"),
            Card("Assassin"),

            Card("Captain"),
            Card("Captain"),
            Card("Captain"),

            Card("Contessa"),
            Card("Contessa"),
            Card("Contessa"),

            Card("Duke"),
            Card("Duke"),
            Card("Duke"),
        ]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop(0)

    def add(self, card):
        self.cards.append(card)