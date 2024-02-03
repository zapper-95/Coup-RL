import random
class Player():

    def __init__(self, id):
        self.id = id


    def get_action(self, state_space, action_space) -> str:
        """Code to get the action of a player given a state
            Currently set to random
        """
        return action_space[random.randint(0,len(action_space)-1)]
    
    def get_id(self):
        return self.id