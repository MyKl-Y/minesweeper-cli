"""
Module: input
Description: Handles keyboard inputs and terminal configuration.
"""

import sys
import tty
import termios
import select
from enum import Enum

class Action(Enum):
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    TOGGLE_FLAG = 5
    REVEAL_CELL = 6
    TOGGLE_SOLVER = 7
    TOGGLE_PROBABILITY = 8
    QUIT = 9
    NEW_GAME = 0

def set_raw_mode():
    """
    Configures the terminal to raw mode for immediate key reading.
    
    Hint: Use tty.setraw and termios to disable input buffering and echo.
    """
    global _original_settings
    _original_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

def restore_terminal():
    """
    Restores the terminal to its default settings.
    
    Hint: Re-enable buffering and echo after the program ends.
    """
    if _original_settings is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_settings)

def get_input() -> str:
    """
    Reads raw input from the terminal (including escape sequences).
    
    Hint: Read character-by-character from sys.stdin.
    """
    char = sys.stdin.read(1)
    if char == '\x1b':
        if select.select([sys.stdin], [], [], 0.01)[0]:
            char += sys.stdin.read(1)
            if select.select([sys.stdin], [], [], 0.01)[0]:
                char += sys.stdin.read(1)
    return char

def parse_input(input_str: str) -> Action:
    """
    Parses the raw input string into an Action.
    
    Hint: Map escape sequences (e.g., arrow keys) and single characters to Actions.
    """
    mapping = {
        "\x1b[A": Action.MOVE_UP,
        "w": Action.MOVE_UP,
        "\x1b[B": Action.MOVE_DOWN,
        "s": Action.MOVE_DOWN,
        "\x1b[D": Action.MOVE_LEFT,
        "a": Action.MOVE_LEFT,
        "\x1b[C": Action.MOVE_RIGHT,
        "d": Action.MOVE_RIGHT,
        "f": Action.TOGGLE_FLAG,
        "r": Action.REVEAL_CELL,
        "S": Action.TOGGLE_SOLVER,
        "P": Action.TOGGLE_PROBABILITY,
        "Q": Action.QUIT,
        "N": Action.NEW_GAME
    }
    return mapping.get(input_str, None)

def get_action_from_input() -> Action:
    """
    High-level function to capture an action from user input.
    
    Hint: Combine get_input() and parse_input() to return a valid Action.
    """
    while True:
        input_str = get_input()
        action = parse_input(input_str)
        if action is not None:
            return action
