from player import Player
from deck import Deck
import random
num_players = 2

cards = ["ambassador", "assassin", "captain", "contessa", "duke"]

state_space = {
    "player_1_card_1": None,
    "player_1_card_2": None,
    "player_2_card_1": None,
    "player_2_card_2": None,
    "player_1_card_1_alive": True,
    "player_1_card_2_alive": True,
    "player_2_card_1_alive": True,
    "player_2_card_2_alive": True,
    "player_1_coins": 1,
    "player_2_coins": 1,
    "player_1_proposed_action": None,
    "player_2_proposed_action": None,
}


action_space = {
    0:"income",
    1:"foreign_aid",
    2:"tax",
    3: "assassinate",
    4: "exchange",
    5: "steal",
    6: "counteract",
    7: "challenge",
    8: "coup",
}

action_card = {
    "tax": "duke",
    "assissinate": "assassin",
    "exchange": "ambassador",
    "steal": "captain",
}

action_counter_card = {
    "foreign_aid":"duke",
    "assassinate":"contessa",
    "steal": ["ambassador", "captain"],   
}

def print_game(turn, state_space):
    """Print the state space in a readable format"""
    print("----------------")
    print(f"Turn: P{turn+1}")
    alive_cards = []
    for i in range(2):
        for j in range(2):
            if state_space[f"player_{i+1}_card_{j+1}_alive"]:
                alive_cards.append(state_space[f"player_{i+1}_card_{j+1}"])
            else:
                alive_cards.append("dead")
    print("----------------")

    print(f"Player 1: {alive_cards[0]} {alive_cards[1]} {state_space['player_1_coins']}")
    print(f"Player 2: {alive_cards[2]} {alive_cards[3]} {state_space['player_2_coins']}")


def end_state(state_space):
    """Check if the game has ended"""
    return ((not state_space["player_1_card_1_alive"] and not state_space["player_1_card_2_alive"])
            or 
            (not state_space["player_2_card_1_alive"] and not state_space["player_2_card_2_alive"]))

def loose_card(player_id, state_space):
    """Loose a card for a player"""
    if(state_space[f"player_{player_id+1}_card_1_alive"]):
        state_space[f"player_{player_id+1}_card_1_alive"] = False
    else:
        state_space[f"player_{player_id+1}_card_2_alive"] = False

def play_action(proposed_action:str, player_id:int, state_space:dict, deck:Deck):
    """Play an action and update the state space"""

    if proposed_action == "income":
        state_space[f"player_{player_id+1}_coins"] += 1
    elif proposed_action == "foreign_aid":
        state_space[f"player_{player_id+1}_coins"] += 2
    elif proposed_action == "tax":
        state_space[f"player_{player_id+1}_coins"] += 3
    elif proposed_action == "assassinate" and state_space[f"player_{player_id+1}_coins"] >= 3:
        loose_card((player_id+1)%2, state_space)
        state_space[f"player_{player_id+1}_coins"] -=3
    elif proposed_action == "exchange":
        deck.add_card(state_space[f"player_{player_id+1}_card_1"])
        deck.add_card(state_space[f"player_{player_id+1}_card_2"])
        state_space[f"player_{player_id+1}_card_1"] = deck.draw_card()
        state_space[f"player_{player_id+1}_card_2"] = deck.draw_card()
    elif proposed_action == "steal":
        state_space[f"player_{player_id+1}_coins"] += min(2, state_space[f"player_{((player_id+1)%2)+1}_coins"])
        state_space[f"player_{((player_id+1)%2)+1}_coins"] -= min(2, state_space[f"player_{((player_id+1)%2)+1}_coins"])
    elif proposed_action == "coup" and state_space[f"player_{player_id+1}_coins"] >= 7:

        state_space[f"player_{(player_id+1)}_coins"] -= 7
        loose_card((player_id+1)%2, state_space)

    else:
        print("invalid action")

        



def action_legal(state, player_id, action):
    """Check if an action of given player is legal for the cards they have"""
    cards = [state[f"player_{player_id+1}_card_1"], state[f"player_{player_id+1}_card_2"]]

    if action in action_card.keys():
        if not action_card[action] in cards:
            return False
        
    return True
    
def can_counteract(action):
    """Check if an action can be counteracted"""
    return action in action_counter_card.keys()


def counteraction_legal(state, player_id, stop_action):
    """Check if a player can stop an action of another"""

    # cards of the counteracting player
    cards = [state[f"player_{player_id+1}_card_1"], state[f"player_{player_id+1}_card_2"]]

    # check that the action they are stopping can be counteracted
    if can_counteract(stop_action):

        # check that they have at least one of the required cards to stop the action
        if len(set(cards).intersection(set(action_counter_card[stop_action]))) > 0:
            return True

    return False


    
    


def main():
    player_turn = 0


    players = [Player(i) for i in range(num_players)]
    deck = Deck(cards)



    for i in range(num_players):
        for j in range(2):
            state_space[f"player_{i+1}_card_{j+1}"] = deck.draw_card()


    while not end_state(state_space):
        print_game(player_turn, state_space)
        #input()
        player_queue = []
        current_player = players[player_turn]

        player_queue.append(current_player.get_id())


        proposed_action = current_player.get_action(state_space, action_space)
        print(proposed_action)

        state_space[f"player_{current_player.get_id()+1}_proposed_action"] = proposed_action


        # iterate through all players to see if they challenge
        for secondary_player in players:

            if secondary_player != current_player:
                second_action = secondary_player.get_action(state_space, action_space)

                # a -> ch
                if second_action == "challenge":
                    print("challenge")
                    player_queue.append(secondary_player.get_id())
                    state_space[f"player_{secondary_player.get_id()+1}_proposed_action"] = "challenge"
                    break

                # a -> co or a -> co -> ch
                elif second_action == "counteract":
                    print("counteract")
                    player_queue.append(secondary_player.get_id())
                    state_space[f"player_{secondary_player.get_id()+1}_proposed_action"] = "counteract"

                    for tertiary_player in players:
                        
                        # a -> ch
                        if tertiary_player != secondary_player and tertiary_player.get_action(state_space, action_space) == "challenge":
                            print("challenge")
                            player_queue.append(tertiary_player.get_id())
                            break
                    break

        

        # if the player queue is only 1, then no counteraction or challenges
        if len(player_queue) == 1:
            play_action(proposed_action, player_queue[0], state_space, deck)
        elif len(player_queue) == 2:
            
            # if a -> co nothing happens
            # if a -> ch then check if action was legal

            if(state_space[f"player_{player_queue[1]+1}_proposed_action"]) == "challenge":   
                if action_legal(state_space, player_queue[0],proposed_action):
                    play_action(proposed_action, player_queue[0], state_space, deck)
                    loose_card(player_queue[1], state_space)
                else:
                    loose_card(player_queue[0], state_space)
            else:
                if not can_counteract(proposed_action):
                    print("invalid counteraction")
                    play_action(proposed_action, player_queue[0], state_space, deck)

        else:
            if counteraction_legal(state_space, player_queue[1], proposed_action):
                # counteraction was valid so challenging player looses card
                loose_card(player_queue[2], state_space)
            else:
                # counteraction was not valid, so action goes through, and counteracting player looses card
                play_action(proposed_action, player_queue[0], state_space, deck)
                loose_card(player_queue[1], state_space)
            # player queue of size 3, so a, co, ch
        
        state_space["player_1_proposed_action"] = None
        state_space["player_2_proposed_action"] = None
        player_turn = (player_turn+1)%2

    print("\n----------------")
    print("game over")
    print("---------------- \n")
    print_game(player_turn, state_space)



if __name__ == '__main__':
    main()