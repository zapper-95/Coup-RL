import coup_v2



def create_environment():
    return coup_v2.env()
    

def test_assassinate():
    env = create_environment()
    env.reset()


    env.state_space["player_1_card_1"] = "assassin"
    env.state_space["player_1_card_2"] = "contessa"
    env.state_space["player_1_coins"] = 3

    env.state_space["player_2_card_1"] = "captain"
    env.state_space["player_2_card_2"] = "duke"

    # attempt to assassinate
    env.step(env.get_action_id("assassinate"))
    env.step(env.get_action_id("kill_card_1"))

    assert env.state_space["player_1_coins"] == 0
    assert env.state_space["player_2_card_1_alive"] == 0
    print("Assassinate test passed")
    

def test_steal():
    env = create_environment()
    env.reset()

    env.state_space["player_1_card_1"] = "captain"
    env.state_space["player_1_card_2"] = "contessa"
    env.state_space["player_1_coins"] = 3

    env.state_space["player_2_card_1"] = "captain"
    env.state_space["player_2_card_2"] = "duke"
    env.state_space["player_2_coins"] = 3

    # attempt to steal
    env.step(env.get_action_id("steal"))

    assert env.state_space["player_1_coins"] == 5
    assert env.state_space["player_2_coins"] == 1
    print("Steal test passed")



def test_tax():
    env = create_environment()
    env.reset()

    env.state_space["player_1_card_1"] = "duke"
    env.state_space["player_1_card_2"] = "contessa"
    env.state_space["player_1_coins"] = 1

    env.state_space["player_2_card_1"] = "captain"
    env.state_space["player_2_card_2"] = "duke"
    env.state_space["player_2_coins"] = 2

    env.step(env.get_action_id("tax"))

    assert env.state_space["player_1_coins"] == 4
    print("Tax test passed")



def test_exchange():
    env = create_environment()
    env.reset()

    deck_top  = [env.deck.peek_card(0), env.deck.peek_card(1)]

    env.state_space["player_1_card_1"] = "duke"
    env.state_space["player_1_card_2"] = "contessa"
    env.state_space["player_1_coins"] = 1

    env.state_space["player_2_card_1"] = "captain"
    env.state_space["player_2_card_2"] = "duke"
    env.state_space["player_2_coins"] = 2

    env.step(env.get_action_id("exchange"))

    assert deck_top[0] == env.state_space["player_1_card_1"]
    assert deck_top[1] == env.state_space["player_1_card_2"]
    print("Exchange test passed")



def counter_action_test():
    actions = ["assassinate", "steal", "foreign_aid"]
    env = create_environment()

    for action in actions:
        env.reset()

        env.state_space["player_1_coins"] = 3

        initial_state = env.state_space.copy()

        env.step(env.get_action_id(action))
        env.step(env.get_action_id("counteract"))
        env.step(env.get_action_id("pass"))


        if action == "assassinate":
            assert env.state_space["player_1_coins"] == 0
            # assert all other state space values are the same
            for key in initial_state:
                if key != "player_1_coins":
                    assert env.state_space[key] == initial_state[key]

        else:
            assert env.state_space == initial_state

        print(f"{action} counteraction test passed")



def correct_challenge_test():
    env = create_environment()
    actions = [action for action in env.action_card.keys()]

    for action in actions:
        env.reset()

        # give the player the card they need to perform the action
        env.state_space["player_1_card_1"] = env.action_card[action]

        # give the player enough coins to assassinate
        env.state_space["player_1_coins"] = 3

        env.step(env.get_action_id(action))
        env.step(env.get_action_id("challenge"))
        env.step(env.get_action_id("pass"))
        env.step(env.get_action_id("kill_card_1"))

        assert env.state_space["player_2_card_1_alive"] == False
        print(f"{action} correct challenge test passed")


def incorrect_challenge_test():
    env = create_environment()
    actions = [action for action in env.action_card.keys()]

    for action in actions:
        env.reset()


        # give the player two contessa's so none of these actions are legal
        env.state_space["player_1_card_1"] = "contessa"
        env.state_space["player_1_card_2"] = "contessa"

        # give the player enough coins to assassinate
        env.state_space["player_1_coins"] = 3

        initial_state = env.state_space.copy()
        env.step(env.get_action_id(action))
        env.step(env.get_action_id("challenge"))
        env.step(env.get_action_id("kill_card_1"))

        # prove the initial state is the same as the final state except for player 1's lost card

        discount_states = ["player_1_card_1_alive"]
        if action == "assassinate": discount_states.append("player_1_coins")
        for key in initial_state:

            
            if key not in discount_states:
                assert env.state_space[key] == initial_state[key]

        print(f"{action} incorrect challenge test passed")




def correct_counteract_challenge():
    env = create_environment()
    actions = [action for action in env.action_counter_card.keys()]

    for action in actions:
        env.reset()
        # give the first player enough coins to assassinate
        env.state_space["player_1_coins"] = 3

        # give the second player only assassins (which cannot counteract anything)
        env.state_space["player_2_card_1"] = "assassin"
        env.state_space["player_2_card_2"] = "assassin"

        

        env.step(env.get_action_id(action))
        player_action_state = env.state_space.copy()
        env.step(env.get_action_id("counteract"))
        env.step(env.get_action_id("challenge"))
        env.step(env.get_action_id("kill_card_1"))

        # prove the final state is equal to just the state after taking the first action (except for the lost card of player 2)
        discount_states = ["player_2_card_1_alive"]

        # since both cards are killed with an assasainate, player 2s second card is also killed
        if action == "assassinate": discount_states = discount_states + ["player_2_card_1_alive", "player_2_card_2_alive", "player_2_loose_card"]
        for key in player_action_state:

            if key not in discount_states:
                assert env.state_space[key] == player_action_state[key]
        print(f"{action} correct counteraction test passed")

def incorrect_counteract_challenge():
    env = create_environment()
    actions = [action for action in env.action_counter_card.keys()]


    for action in actions:

        env.reset()
        # give the first player enough coins to assassinate
        env.state_space["player_1_coins"] = 3

        # give the second player the card to counteract the action
        counter_card = env.action_counter_card[action]
        
        if type(counter_card) == list:
            # if there are two cards that can counter, only give one
            counter_card = counter_card[0]

        env.state_space["player_2_card_1"] = counter_card
        
        initial_state = env.state_space.copy()
        top_deck_card = env.deck.peek_card(0)

        env.step(env.get_action_id(action))
        env.step(env.get_action_id("counteract"))
        env.step(env.get_action_id("challenge"))
        env.step(env.get_action_id("pass"))
        env.step(env.get_action_id("kill_card_1"))

        discount_states = ["player_1_card_1_alive"]
        if action == "assassinate": discount_states.append("player_1_coins")
        # prove the initial state is the same as the final state except for player 1's lost card
        for key in initial_state:
            if key == "player_2_card_1":
                # ensure the players proven card is swapped with the top deck card
                assert env.state_space[key] == top_deck_card
            elif key not in discount_states:
                assert env.state_space[key] == initial_state[key]
        print(f"{action} incorrect counteraction test passed")




test_assassinate()
test_steal()
test_tax()
test_exchange()
counter_action_test()
correct_challenge_test()
incorrect_challenge_test()
correct_counteract_challenge()
incorrect_counteract_challenge()

    