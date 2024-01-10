import random
class Player(object):

    def __init__(self, name, deck):
        self.name = name

        # 2 face down cards
        self.card_1 = deck.draw()
        self.card_2 = deck.draw()

        self.coins = 1

    def get_coins(self) -> int:
        return self.coins

    def assasinate(self, other_player):
        if self.coins >= 3:
            self.coins -= 3
            other_player.lose_card()

    def income(self):
        self.coins += 1

    def foreign_aid(self):
        self.coins += 2

    def tax(self):
        self.coins += 3


    # TODO: Current swaps all cards. Should be changed so this is a choice
    def exchange(self, deck):
        if not self.card_1.dead:
            deck.add(self.card_1)
            self.card_1 = deck.draw()
        
        if not self.card_2.dead:
            deck.add(self.card_2)
            self.card_2 = deck.draw()

            

            


    def steal(self, other_player):
        if other_player.coins >= 2:
            self.coins += 2
            other_player.coins -= 2
        else:
            self.coins += other_player.coins
            other_player.coins = 0   



    def coup(self, other_player):
        if self.coins >= 7:
            self.coins -= 7
            other_player.lose_card()



    #TODO: Change this to choose which card to lose
    def lose_card(self):
        if not self.card_1.is_dead():
            self.card_1.kill()
        else:
            self.card_2.kill()

