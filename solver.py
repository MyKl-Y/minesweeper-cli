"""
Module: solver
Description: Contains logic for a Minesweeper solver that suggests safe moves.
"""

import numpy as np
from board import update_board
from typing import List, Tuple, Set
from ms_types import Board, GameState, Mode, Cell

def check_win_condition(board: Board, num_mines: int) -> bool:
    """
    Checks if the win condition is met.
    
    Hint: The win condition is met when all non-mine cells are revealed.
    """
    rows, cols = len(board), len(board[0])

    revealed_cells = 0
    not_revealed_cells = 0
    flagged_mines = 0

    for row in range(rows):
        for col in range(cols):
            cell = board[row][col]
            if cell.is_revealed:
                revealed_cells += 1
            elif cell.is_flagged:
                flagged_mines += 1
            else:
                not_revealed_cells += 1

    potential_found_mines = not_revealed_cells + flagged_mines
    if (flagged_mines + not_revealed_cells == num_mines) and (revealed_cells + potential_found_mines == len(board) * len(board[0])):
        return True
    return False

def solve_board(board: Board, method: str) -> List[Tuple[int, int]]:
    """
    Analyzes the board and returns a list of safe cell positions to reveal.
    
    Hint: Use Minesweeper strategies to deduce safe moves.
    """
    rows, cols = len(board), len(board[0])

    if method == "set_theory":
        constraints: List[Tuple[Set[Tuple[int, int]], int]] = []

        def get_adjacent_positions(board: Board, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            rows, cols = len(board), len(board[0])
            positions = set()
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    new_row, new_col = pos[0] + i, pos[1] + j
                    if 0 <= new_row < rows and 0 <= new_col < cols:
                        positions.add((new_row, new_col))
            return positions
        
        for row in range(rows):
            for col in range(cols):
                cell: Cell = board[row][col]
                if cell.is_revealed and cell.adjacent_count > 0:
                    adjacent_positions = get_adjacent_positions(board, (row, col))
                    flagged_count = 0
                    unknown_positions: Set[Tuple[int, int]] = set()
                    for pos in adjacent_positions:
                        r2, c2 = pos
                        if board[r2][c2].is_flagged:
                            flagged_count += 1
                        elif not board[r2][c2].is_revealed:
                            unknown_positions.add(pos)
                    mines_remaining = cell.adjacent_count - flagged_count
                    if unknown_positions:
                        constraints.append((unknown_positions, mines_remaining))
        changes = True
        while changes:
            changes = False
            new_constraints: List[Tuple[Set[Tuple[int, int]], int]] = []
            for s1, n1 in constraints:
                for s2, n2 in constraints:
                    if s1 == s2:
                        continue
                    if s1.issubset(s2):
                        new_s = s2 - s1
                        new_n = n2 - n1
                        if (new_s, new_n) not in constraints and (new_s, new_n) not in new_constraints:
                            new_constraints.append((new_s, new_n))
                            changes = True
                    elif s2.issubset(s1):
                        new_s = s1 - s2
                        new_n = n1 - n2
                        if (new_s, new_n) not in constraints and (new_s, new_n) not in new_constraints:
                            new_constraints.append((new_s, new_n))
                            changes = True
            constraints.extend(new_constraints)

        moves: Set[Tuple[int, int]] = set()

        for s, n in constraints:
            if n == 0:
                moves.update(s)

        return list(moves)
    elif method == "matrix":
        def get_neighbors(board: Board, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            rows, cols = len(board), len(board[0])
            neighbors = []
            for i in range(-1, 0, 1):
                for j in range(-1, 0, 1):
                    if 0 == i and 0 == j:
                        continue
                    new_row, new_col = pos[0] + i, pos[1] + j
                    if 0 <= new_row < rows and 0 <= new_col < cols:
                        neighbors.append((new_row, new_col))
            return neighbors

        def build_equations(board: Board):
            rows, cols = board.shape
            active_squares = []  # List of tuples: ((i, j), list of unknown neighbors, clue)
            
            # Identify active squares.
            for i in range(rows):
                for j in range(cols):
                    cell: Cell = board[i][j]
                    if cell.is_revealed:  # revealed number
                        neigh = get_neighbors(board, (i, j))
                        unknown_neighbors = []
                        flagged_count = 0
                        for (ni, nj) in neigh:
                            neigh_cell: Cell = board[ni][nj]
                            if neigh_cell.is_flagged:
                                flagged_count += 1
                            elif not neigh_cell.is_revealed:
                                unknown_neighbors.append((ni, nj))
                        if unknown_neighbors:
                            clue = cell.adjacent_count - flagged_count
                            active_squares.append(((i, j), unknown_neighbors, clue))
            
            # Map every unknown square (adjacent to an active square) to a unique index.
            var_index = {}
            index = 0
            for (_, unknown_list, _) in active_squares:
                for pos in unknown_list:
                    if pos not in var_index:
                        var_index[pos] = index
                        index += 1
                        
            # Build the list of constraint equations.
            equations = []
            for (_, unknown_list, clue) in active_squares:
                eq_vars = set(var_index[pos] for pos in unknown_list)
                equations.append((eq_vars, clue))
            
            return equations, var_index

        def solve_equations(equations):
            assignments = {}
            changed = True
            while changed:
                changed = False
                for (vars_set, target) in equations:
                    # Sum up values already assigned in this equation.
                    assigned_sum = sum(assignments.get(v, 0) for v in vars_set if v in assignments)
                    # List the variables not yet assigned.
                    unsolved = [v for v in vars_set if v not in assignments]
                    remaining = target - assigned_sum
                    # If all unsolved variables must be mines:
                    if unsolved and remaining == len(unsolved):
                        for v in unsolved:
                            if v not in assignments:
                                assignments[v] = 1
                                changed = True
                    # If none of the unsolved variables can be mines:
                    elif unsolved and remaining == 0:
                        for v in unsolved:
                            if v not in assignments:
                                assignments[v] = 0
                                changed = True
            return assignments

        # TODO: Fix the solver to work with the matrix method.
        equations, var_index = build_equations(board)
        assignments = solve_equations(equations)
        
        moves = []

        index_to_pos = {v: pos for pos, v in var_index.items()}
        for var, value in assignments.items():
            pos = index_to_pos[var]
            if value == 0:
                moves.append(pos)
        return moves
    elif method == "probability":
        moves = []
        for row in range(rows):
            for col in range(cols):
                cell = board[row][col]
                if not cell.is_revealed:
                    if cell.probability == 0:
                        moves.append((row, col))
        return moves
    else:
        moves = []

        for row in range(rows):
            for col in range(cols):
                cell = board[row][col]
                if not cell.has_mine and not cell.is_revealed:
                    moves.append((row, col))
        return moves
        # raise ValueError("Invalid method provided. Choose from 'set_theory', 'matrix', or 'probability'.")



def apply_solver(state: GameState, method: str, mines: int) -> GameState:
    """
    Applies the solver's recommendation to the game state.
    
    Hint: Automatically reveal cells that are determined safe.
    """
    iterations = 0

    while (not check_win_condition(state.board, mines)) and (iterations < mines):
        board = state.board
        safe_moves = solve_board(board, method)

        for move in safe_moves:
            state.board = update_board(board, move)

        iterations += 1

    return state