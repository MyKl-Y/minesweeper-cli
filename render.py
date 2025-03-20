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
            cell_str = draw_cell((i, j), cell, is_cursor, state)
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

def draw_cell(pos: tuple, cell: Cell, is_cursor: bool, state: GameState) -> str:
    """
    Draws an individual cell at a given position.
    
    Hint: Use print statements and ANSI escape codes to position the cursor and draw symbols.
    """
    if state.win and cell.has_mine:
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
    elif not cell.is_revealed and not state.win:
        """if state.game_mode == Mode.PROB:
            color_index = 0
            if cell.probability >= 0.8:
                color_index = 1
            elif cell.probability >= 0.5:
                color_index = 2
            elif cell.probability >= 0.3:
                color_index = 3
            elif cell.probability >= 0.1:
                color_index = 4
            symbol = f"{cell_colors[color_index]}{cell.probability}{cell_colors['END']}"
        else:"""
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
        print("\r\x1b[93mMine probabilities:\x1b[0m")
        print("\r\x1b[93m-------------------\x1b[0m")
        for i in range(len(state.board)):
            row_str = "\r"
            for j in range(len(state.board[0])):
                cell: Cell = state.board[i][j]
                if not cell.is_revealed:
                    if cell.probability is not None:
                        if cell.probability == 1.0:
                            row_str += f"\x1b[41m100\x1b[0m "
                        else:
                            row_str += f"{int(round(cell.probability*100, 0)):03d} "
                else:
                    if cell.is_revealed:
                        if cell.has_mine:
                            symbol = f" {cell_colors['*']} * {cell_colors['END']}"
                        else:
                            count = cell.adjacent_count
                            symbol = f" {cell_colors[count]}{count}{cell_colors['END']}"
                    elif cell.is_flagged:
                        symbol = f" {cell_colors['F']} F {cell_colors['END']}"
                    row_str += symbol + "  "
            print(row_str)
    elif state.win:
        print("\r\x1b[32mCongratulations\x1b[0m! You won!")
    elif state.game_over:
        print("\r\x1b[31mGame over\x1b[0m! You hit a mine!")
    else:
        print(f"\r{cell_colors['*']}{state.num_mines}{cell_colors['END']} mines | {cell_colors[3]}{state.num_flags}{cell_colors['END']} flags")