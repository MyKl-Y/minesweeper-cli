"""
Module: board
Description: Functions for initializing and updating the Minesweeper board.
"""

import numpy as np
from typing import Tuple
from ms_types import Cell, Board

def init_board(rows: int, cols: int) -> Board:
    """
    Initializes an empty board with the specified number of rows and columns.
    
    Hint: Create a 2D list of Cell objects with default values (no mine, not revealed, not flagged).
    """
    return np.array([[Cell(False, False, False, 0, 0.0) for _ in range(cols)] for _ in range(rows)], dtype=Cell)

def place_mines(board: Board, num_mines: int) -> Board:
    """
    Randomly places a given number of mines on the board.
    
    Hint: Use random.choice or random.sample or np.random to choose positions, ensuring no duplicates.
    """
    rows, cols = len(board), len(board[0])
    mine_positions = np.random.choice(rows * cols, num_mines, replace=False)
    for pos in mine_positions:
        row, col = pos // cols, pos % cols
        board[row][col].has_mine = True
    for i in range(rows):
        for j in range(cols):
            if not board[i][j].has_mine:
                board[i][j].adjacent_count = count_adjacent_mines(board, (i, j))
    return board

def place_mines_after_first_move(board: Board, num_mines: int, first_move: Tuple[int, int]) -> Board:
    """
    Places mines on the board after the first move, ensuring that the cell at `first_pos` 
    and its immediate neighbors are free of mines.
    
    Hint:
        - Determine a 'safe zone' around the first move (e.g., first cell and its 8 neighbors).
        - Exclude these positions from available mine positions.
        - Use np.random.choice on the remaining positions.
        - Finally, update adjacent counts for all cells.
    """
    rows, cols = len(board), len(board[0])
    safe_zone = set()
    for i in range(-1, 2):
        for j in range(-1, 2):
            row, col = first_move[0] + i, first_move[1] + j
            if 0 <= row < rows and 0 <= col < cols:
                safe_zone.add(row * cols + col)
    all_positions = set(range(rows * cols))
    available_positions = list(all_positions - safe_zone)
    mine_positions = np.random.choice(available_positions, num_mines, replace=False)
    for pos in mine_positions:
        row, col = pos // cols, pos % cols
        board[row][col].has_mine = True
    for i in range(rows):
        for j in range(cols):
            if not board[i][j].has_mine:
                board[i][j].adjacent_count = count_adjacent_mines(board, (i, j))
    return board

def update_board(board: Board, pos: Tuple[int, int]) -> Board:
    """
    Updates the board based on a move at the specified position.
    
    Hint: Reveal the cell at 'pos' and, if empty, perform a flood fill to reveal adjacent cells.
    """
    def reveal_cell(board, pos):
        if not 0 <= pos[0] < len(board) or not 0 <= pos[1] < len(board[0]) or board[pos[0]][pos[1]].is_revealed:
            return
        board[pos[0]][pos[1]].is_revealed = True
        if board[pos[0]][pos[1]].adjacent_count == 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    reveal_cell(board, (pos[0] + i, pos[1] + j))
    reveal_cell(board, pos)
    return board

def count_adjacent_mines(board: Board, pos: Tuple[int, int]) -> int:
    """
    Counts the number of mines adjacent to the cell at the given position.
    
    Hint: Check the eight surrounding cells (take care of board boundaries).
    """
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            row, col = pos[0] + i, pos[1] + j
            if 0 <= row < len(board) and 0 <= col < len(board[0]) and board[row][col].has_mine:
                count += 1
    return count
