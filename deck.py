import random
from card import Card, Ambassador, Assassin, Captain, Contessa, Duke

class Deck(object):
    def __init__(self):
        self._cards = [
            Ambassador(),
            Ambassador(),
            Ambassador(),

            Assassin(),
            Assassin(),
            Assassin(),

            Captain(),
            Captain(),
            Captain(),

            Contessa(),
            Contessa(),
            Contessa(),

            Duke(),
            Duke(),
            Duke(),
        ]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self._cards)

    def draw(self):
        return self._cards.pop(0)

    def add(self, card:Card):
        self._cards.append(card)