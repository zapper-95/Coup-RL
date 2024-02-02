from action import Assassinate, Exchange, Steal, Tax, ForeignAid
from abc import ABC, abstractmethod
class Card(ABC):

    def __init__(self, name):
        self._name = name
        self._dead = False
        
    def kill(self) -> None:
        self._dead = True

    def get_name(self) -> str:
        return self._name

    def is_dead(self) -> bool:
        return self._dead 


class Duke(Card):

    def __init__(self):
        super(Duke, self).__init__("Duke")
        self._name = "Duke"
        self._action = Tax()
        self._counteracts = ["Foreign Aid"]
    
    def action(self):
        return self._action
    
    def counteracts(self):
        return self._counteracts
    

class Captain(Card):
    def __init__(self):
        super(Captain, self).__init__("Captain")
        self._action = Steal()
        self._counteracts = ["Steal"]
    
    def action(self):
        return self._action
    
    def counteracts(self):
        return self._counteracts
    

class Ambassador(Card):
    def __init__(self):
        super(Ambassador, self).__init__("Ambassador")
        self._action = Exchange()
        self._counteracts = ["Steal"]
    
    def action(self):
        return self._action
    
    def counteracts(self):
        return self._counteracts
    

class Assassin(Card):
    def __init__(self):
        super(Assassin, self).__init__("Assassin")
        self._action = Assassinate()
        self._counteracts = []

    def action(self):
        return self._action
    
    def counteracts(self):
        return self._counteracts
    

class Contessa(Card):
    def __init__(self):
        super(Contessa, self).__init__("Contessa")
        self._action = None
        self._counteracts = ["Assassinate"]

    def action(self):
        return self._action
    
    def counteracts(self):
        return self._counteracts
