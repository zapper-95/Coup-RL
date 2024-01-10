class Card(object):

    def __init__(self, name):
        self.name = name
        self.dead = False

    
    def kill(self):
        self.dead = True

    def get_name(self) -> str:
        return self.name

    def is_dead(self) -> bool:
        return self.dead    



