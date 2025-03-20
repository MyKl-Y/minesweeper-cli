"""
Module: render
Description: Provides functions for rendering the game board and UI in the terminal.
"""

import os
import sys
import numpy as np
from ms_types import GameState, Cell, Mode

cell_colors = {
    "F": '\033[91m',
    "*": '\033[91m',
    "win": '\033[42m',
    0: '\033[90m',
    1: '\033[94m',
    2: '\033[92m',
    3: '\033[93m',
    4: '\033[95m',
    5: '\033[96m',
    6: '\033[97m',
    7: '\033[91m',
    8: '\033[92m',
    "END": '\033[0m'
}

def render_state(state: GameState):
    """
    Clears the terminal screen and renders the entire game state.
    
    Hint: Use os.system('clear') to refresh the display, then iterate through the board.
    """
    os.system('clear')

    board: np.ndarray = state.board
    rows, cols = len(board), len(board[0])

    render_additional_info(state)

    for i in range(rows):
        state_str = "\r"
        for j in range(cols):
            is_cursor = (i, j) == state.cursor_pos
            cell: Cell = board[i][j]
            cell_str = draw_cell((i, j), cell, is_cursor, state.win)
            state_str += cell_str + " "
        print(state_str)

    if state.win or state.game_over:
        print("\n\rUse:")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mn\033[0m' ['\033[94mN\033[0m'] to start new game")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mq\033[0m' ['\033[94mQ\033[0m'] to quit.\n")
    else:
        print("\n\rUse:")
        print("\r - '\033[94mw\033[0m', '\033[94ma\033[0m', '\033[94ms\033[0m', and/or '\033[94md\033[0m' to move")
        print("\r - '\033[94mr\033[0m' to reveal")
        print("\r - '\033[94mf\033[0m' to flag")
        print("\r - '\033[93mSHIFT\033[0m + \033[94ms\033[0m' ['\033[94mS\033[0m'] to toggle solver")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mp\033[0m' ['\033[94mP\033[0m'] to toggle probability")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mq\033[0m' ['\033[94mQ\033[0m'] to quit.\n")

def draw_cell(pos: tuple, cell: Cell, is_cursor: bool, win: bool) -> str:
    """
    Draws an individual cell at a given position.
    
    Hint: Use print statements and ANSI escape codes to position the cursor and draw symbols.
    """
    if win and cell.has_mine:
        if cell.is_flagged:
            symbol = f"{cell_colors['win']}F{cell_colors['END']}"
        else:
            symbol = f"{cell_colors['win']}*{cell_colors['END']}"
    if cell.is_revealed:
        if cell.has_mine:
            symbol = f"{cell_colors['*']}*{cell_colors['END']}"
        else:
            count = cell.adjacent_count
            symbol = f"{cell_colors[count]}{count}{cell_colors['END']}"
    elif cell.is_flagged:
        symbol = f"{cell_colors['F']}F{cell_colors['END']}"
    elif not cell.is_revealed and not win:
        symbol = "□"

    if is_cursor:
        symbol = f"\x1b[41m\033[4m{symbol}\033[0m" # Underline the cursor cell

    return symbol

def render_additional_info(state: GameState):
    """
    Renders any additional information (like mine probabilities) based on the current game mode.
    
    Hint: When in PROB mode, display probability information near each cell.
    """
    if state.game_mode == Mode.PROB:
        board = state.board
        rows, cols = board.shape

        for i in range(rows):
            line = ""
            for j in range(cols):
                cell = board[i][j]
                if not cell.is_revealed:
                    prob = f"{cell_colors['F']}P{cell_colors['END']}"
                else:
                    prob = " "
                line += prob + " "
            print(line)
    if state.win:
        print("\r\x1b[32mCongratulations\x1b[0m! You won!")
    elif state.game_over:
        print("\r\x1b[31mGame over\x1b[0m! You hit a mine!")
    else:
        print(f"\r{cell_colors['*']}{state.num_mines}{cell_colors['END']} mines | {cell_colors[3]}{state.num_flags}{cell_colors['END']} flags")