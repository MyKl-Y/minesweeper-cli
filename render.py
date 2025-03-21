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
    "FLAG": '\033[47;91m',
    "*": '\033[40;91m',
    "win": '\033[42m',
    0: '\033[40;90m',
    1: '\033[40;94m',
    2: '\033[40;92m',
    3: '\033[40;93m',
    4: '\033[40;95m',
    5: '\033[40;96m',
    6: '\033[40;97m',
    7: '\033[40;91m',
    8: '\033[40;92m',
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
            state_str += cell_str
        print(state_str)

    if state.win or state.game_over:
        print("\n\rUse:")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mn\033[0m' ['\033[94mN\033[0m'] to start new game")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mq\033[0m' ['\033[94mQ\033[0m'] to quit.")
    else:
        print("\n\rUse:")
        print("\r - '\033[94mw\033[0m', '\033[94ma\033[0m', '\033[94ms\033[0m', and/or '\033[94md\033[0m' to move")
        print("\r - '\033[94mr\033[0m' to reveal")
        print("\r - '\033[94mf\033[0m' to flag")
        print("\r - '\033[93mSHIFT\033[0m + \033[94ms\033[0m' ['\033[94mS\033[0m'] to toggle solver")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mp\033[0m' ['\033[94mP\033[0m'] to toggle probability")
        print("\r - '\033[93mSHIFT\033[0m + \033[94mq\033[0m' ['\033[94mQ\033[0m'] to quit.")

def draw_cell(pos: tuple, cell: Cell, is_cursor: bool, state: GameState) -> str:
    """
    Draws an individual cell at a given position.
    
    Hint: Use print statements and ANSI escape codes to position the cursor and draw symbols.
    """
    if state.game_mode == Mode.PROB:
        if cell.is_revealed:
            if cell.has_mine:
                symbol = f"{cell_colors['*']}  *  {cell_colors['END']}"
            else:
                count = cell.adjacent_count
                symbol = f"{cell_colors[count]}  {count}  {cell_colors['END']}"
        elif cell.is_flagged:
            symbol = f"{cell_colors['F']}  F  {cell_colors['END']}"
        elif not cell.is_revealed and not state.win:
            color = ""
            symbol = ""
            if cell.probability >= 0.75:
                color = "\x1b[44m"
            elif cell.probability >= 0.50:
                color = "\x1b[42m"
            elif cell.probability >= 0.25:
                color = "\x1b[43m"
            elif cell.probability >= 0:
                color = "\x1b[41m"
            if cell.probability is not None:
                if cell.probability == 0.0:
                    symbol += f"\x1b[42m  0  \x1b[0m"
                elif cell.probability == 1.0:
                    symbol += f"\x1b[41m  X  \x1b[0m"
                else:
                    val = 100 - int(round(cell.probability*100, 0))
                    symbol += f"{color} {val}.{"0" if len(str(val)) == 1 else ""} {cell_colors['END']}"
    else:
        if cell.is_revealed:
            if cell.has_mine:
                symbol = f"{cell_colors['*']}* {cell_colors['END']}"
            else:
                count = cell.adjacent_count
                symbol = f"{cell_colors[count]}{count} {cell_colors['END']}"
        elif cell.is_flagged:
            symbol = f"{cell_colors['FLAG']}F {cell_colors['END']}"
        elif not cell.is_revealed and not state.win:
            symbol = f"\x1b[40;37m▒▒\x1b[0m"

    if state.win and cell.has_mine:
        if cell.is_flagged:
            symbol = f"{cell_colors['win']}F{cell_colors['END']}"
        else:
            symbol = f"{cell_colors['win']}*{cell_colors['END']}"

    if is_cursor:
        symbol = f"\033[1;7;2m{symbol}\033[0m" # Underline the cursor cell

    return symbol

def render_additional_info(state: GameState):
    """
    Renders any additional information (like mine probabilities) based on the current game mode.
    
    Hint: When in PROB mode, display probability information near each cell.
    """
    print("\r\x1b[36mMyne-Sweeper\x1b[0m")
    if state.game_mode == Mode.PROB:
        print("\r\x1b[93mMine probabilities:\x1b[0m")
        '''print("\r\x1b[93m-------------------\x1b[0m")
        for i in range(len(state.board)):
            row_str = "\r"
            for j in range(len(state.board[0])):
                cell: Cell = state.board[i][j]
                if cell.is_flagged:
                    row_str += f" {cell_colors['F']} F {cell_colors['END']} "
                elif not cell.is_revealed:
                    if cell.probability is not None:
                        if cell.probability == 0.0:
                            row_str += f"  \x1b[42m0\x1b[0m  "
                        elif cell.probability == 1.0:
                            row_str += f"  \x1b[41mX\x1b[0m  "
                        else:
                            row_str += f" {100-int(round(cell.probability*100, 0))}. "
                else:
                    if cell.is_revealed:
                        if cell.has_mine:
                            symbol = f"  {cell_colors['*']} * {cell_colors['END']}"
                        else:
                            count = cell.adjacent_count
                            symbol = f"  {cell_colors[count]}{count}{cell_colors['END']}"
                    elif cell.is_flagged:
                        symbol = f"  {cell_colors['F']} F {cell_colors['END']}"
                    row_str += symbol + "  "
            print(row_str)'''
    elif state.win:
        print("\r\x1b[32mCongratulations\x1b[0m! You won!")
    elif state.game_over:
        print("\r\x1b[31mGame over\x1b[0m! You hit a mine!")
    else:
        print(f"\r\x1b[91m{state.num_mines}{cell_colors['END']} mines | {cell_colors[3]}{state.num_flags}{cell_colors['END']} flags")