from abc import ABC, abstractmethod 

class Action(ABC):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def execute(self, player):
        pass

    def get_name(self):
        return self._name

    def can_be_challenged(self):
        return self._can_be_challenged

    def can_be_counteracted(self):
        return self._can_be_counteracted


class Assassinate(Action):
    def __init__(self):
        super().__init__("Assasinate")
        self._can_be_challenged = True
        self._can_be_counteracted = True

    def execute(self, player, other_player):
        if player.get_coins() >= 3:
            player.add_coins(-3)
            other_player.lose_card()


class Income(Action):
    def __init__(self):
        super().__init__("Income")
        self._can_be_challenged = False
        self._can_be_counteracted = False
    
    def execute(self, player):
        player.add_coins(1)


class ForeignAid(Action):
    def __init__(self):
        super().__init__("Foreign Aid")
        self._can_be_challenged = False
        self._can_be_counteracted = True

    def execute(self, player):
        player.add_coins(2)


class Tax(Action):
    def __init__(self):
        super(Tax, self).__init__("Tax")
        self._can_be_challenged = True
        self._can_be_counteracted = False 

    def execute(self, player):
        player.add_coins(3)


class Exchange(Action):
    def __init__(self):
        super().__init__("Exchange")
        self._can_be_challenged = True
        self._can_be_counteracted = False

    def execute(self, player, deck):
        # TODO: Current swaps both cards. Should be changed so this is a choice
        if not player.get_card_1().is_dead():
            deck.add(player.get_card_1())
            player.set_card_1(deck.draw())
        
        if not player.get_card_2().is_dead():
            deck.add(player.get_card_2())
            player.set_card_2(deck.draw())

class Steal(Action):
    def __init__(self):
        super().__init__("Steal")
        self._can_be_challenged = True
        self._can_be_counteracted = True

    def execute(self, player, other_player):
        coins_gained = min(2, other_player.get_coins())
        player.add_coins(coins_gained)
        other_player.add_coins(-coins_gained)
    

class Coup(Action):
    def __init__(self):
        super().__init__("Coup")
        self._can_be_challenged = False
        self._can_be_counteracted = False
    
    def execute(self, player, other_player):
        if player.get_coins() >= 7:
            player.add_coins(-7)
            other_player.lose_card()
        
    





    