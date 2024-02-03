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

@pytest.mark.parametrize("player", [1, 2])
def test_tax(state_space, player) -> None:
    """Tests the state changes correctly when a player takes tax"""
    new_state  = state_space.copy()
    play_action("tax", (player-1), new_state)

    assert new_state[f"player_{player}_coins"] == state_space[f"player_{player}_coins"] + 3


@pytest.mark.parametrize("player, other", [(1,2), (2,1)])
def test_steal(state_space, player, other) -> None:
    """Tests the state changes correctl when a player steals from another player"""
    new_state  = state_space.copy()

    play_action("steal", (player-1), new_state)
    assert new_state[f"player_{player}_coins"] == state_space[f"player_{player}_coins"] + 2
    assert new_state[f"player_{other}_coins"] == state_space[f"player_{other}_coins"] - 2


@pytest.mark.parametrize("player, other", [(1,2), (2,1)])
def test_assissinate(state_space, player, other) -> None:
    """Tests the state changes correctly when a player assissinates another player"""
    state_space[f"player_{player}_coins"] = 3

    new_state  = state_space.copy()


    play_action("assissinate", player-1, new_state)

    assert new_state[f"player_{player}_coins"] == (state_space[f"player_{player}_coins"] - 3)
    assert new_state[f"player_{other}_card_1_alive"] == False or new_state[f"player_{other}_card_2_alive"] == False

@pytest.mark.parametrize("player", [1,2])
def test_foreign_aid(state_space, player) -> None:
    """Tests the state changes correctly when a player takes foreign aid"""
    new_state  = state_space.copy()
    play_action("foreign_aid", player-1, new_state)

    assert new_state[f"player_{player}_coins"] == state_space[f"player_{player}_coins"] + 2 

@pytest.mark.parametrize("player", [1,2])
def test_income(state_space, player) -> None:
    """Tests the state changes correctly when a player takes income"""
    new_state  = state_space.copy()
    play_action("income", player-1, new_state)

    assert new_state[f"player_{player}_coins"] == state_space[f"player_{player}_coins"] + 1

@pytest.mark.parametrize("player, other", [(1,2), (2,1)])
def test_coup(state_space, player, other) -> None:
    """Tests the state changes correctly when a player takes coup"""

    state_space[f"player_{player}_coins"] = 7
    new_state  = state_space.copy()
    play_action("coup", player-1, new_state)

    assert new_state[f"player_{player}_coins"] == state_space[f"player_{player}_coins"] - 7
    assert new_state[f"player_{other}_card_1_alive"] == False or new_state[f"player_{other}_card_2_alive"] == False

    
