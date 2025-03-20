"""
Module: ms_types
Description: Contains common data types for the Minesweeper game.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

class Mode(Enum):
    REVEAL = 1    # Standard mode for revealing cells.
    FLAG = 2      # Mode for flagging/unflagging cells.
    SOLVER = 3    # Mode that enables the solver to suggest moves.
    PROB = 4      # Mode to display mine probabilities.

@dataclass
class Cell:
    has_mine: bool         # True if the cell contains a mine.
    is_revealed: bool      # True if the cell has been revealed.
    is_flagged: bool       # True if the cell is flagged.
    adjacent_count: int    # Number of adjacent mines.

# Define Board as a list of lists of Cells.
Board = np.ndarray

@dataclass
class GameState:
    board: Board           # The current state of the Minesweeper board.
    cursor_pos: Tuple[int, int]  # (row, column) position of the cursor.
    game_mode: Mode        # The current game mode.
    game_over: bool        # True if the game has ended.
    win: bool              # True if the player has won the game.
    num_mines: int         # Number of mines on the board.
    num_flags: int         # Number of flags placed.
    exit: bool             # True if the player chooses to exit the game.
