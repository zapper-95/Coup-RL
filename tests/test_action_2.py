from coup_new import play_action 
import pytest
import copy
"""
pytest will run all files of the
form test_*.py or *_test.py in the current directory and its subdirectories.
"""

@pytest.fixture
def state_space():
    return {
        "player_1_card_1": "ambassador",
        "player_1_card_2": "duke",
        "player_2_card_1": "captain",
        "player_2_card_2": "contessa",
        "player_1_card_1_alive": True,
        "player_1_card_2_alive": True,
        "player_2_card_1_alive": True,
        "player_2_card_2_alive": True,
        "player_1_coins": 1,
        "player_2_coins": 1,
        "player_1_proposed_action": None,
        "player_2_proposed_action": None,
    }


@pytest.parametrize()
def test_steal(state_space, request) -> None:
    """Tests the state changes correctl when a player steals from another player"""

    state_space["player_2_coins"] = request.params[0]

    new_state  = state_space.copy()
    play_action("steal", 0, new_state)
    assert new_state["player_1_coins"] == state_space["player_1_coins"] + request.parmas[1]
    assert new_state["player_2_coins"] == state_space["player_2_coins"] - request.money[1]

def test_assissinate(state_space) -> None:
    """Tests the state changes correctly when a player assissinates another player"""
    state_space["player_1_coins"] = 3

    new_state  = state_space.copy()


    play_action("assissinate", 0, new_state)

    assert new_state["player_1_coins"] == (state_space["player_1_coins"] - 3)
    assert new_state["player_2_card_1_alive"] == False or new_state["player_2_card_2_alive"] == False

def test_foreign_aid(state_space) -> None:
    """Tests the state changes correctly when a player takes foreign aid"""
    new_state  = state_space.copy()
    play_action("foreign_aid", 0, new_state)

    assert new_state["player_1_coins"] == state_space["player_1_coins"] + 2 

def test_income(state_space) -> None:
    """Tests the state changes correctly when a player takes income"""
    new_state  = state_space.copy()
    play_action("income", 0, new_state)

    assert new_state["player_1_coins"] == state_space["player_1_coins"] + 1

def test_coup(state_space) -> None:
    """Tests the state changes correctly when a player takes coup"""

    state_space["player_1_coins"] = 7
    new_state  = state_space.copy()
    play_action("coup", 0, new_state)

    assert new_state["player_1_coins"] == state_space["player_1_coins"] - 7
    assert new_state["player_2_card_1_alive"] == False or new_state["player_2_card_2_alive"] == False

    
