"""
Module: probabilities
Description: Computes and provides mine probability estimates for unrevealed cells.
"""

import numpy as np
from typing import List, Optional, Tuple
from ms_types import Board, GameState, Cell

def calc_probabilities(state: GameState) -> List[List[Optional[float]]]:
    """
    Calculates the probability that each unrevealed cell contains a mine.
    
    Hint: Use heuristics based on adjacent revealed cells to calculate probability.
    """
    board: Board = state.board
    rows, cols = len(board), len(board[0])

    """
        Find tiles with the following properties:
        - They always hold a specific number of mines.
        - They dont influence other mine placements.
        - They are not revealed.
        Also, find floating tiles.
    """
    tiles_with_fixed_mines = []
    tiles_dont_influence = []

    # Identify constrained cells: unrevealed cells adjacent to a revealed cell.
    constrained_set = set()
    for i in range(rows):
        for j in range(cols):
            if board[i][j].is_revealed:
                for pos in get_adjacent_positions(board, (i, j)):
                    r, c = pos
                    if not board[r][c].is_revealed:
                        constrained_set.add(pos)

    # Unconstrained cells: unrevealed cells not adjacent to any revealed cells.
    unconstrained_set = set()
    for i in range(rows):
        for j in range(cols):
            if not board[i][j].is_revealed:
                if (i, j) not in constrained_set:
                    unconstrained_set.add((i, j))

    # Total floating tiles = all unrevealed cells
    total_floating = len(constrained_set) + len(unconstrained_set)
    num_remaining_mines = state.num_mines - state.num_flags
    mine_density = num_remaining_mines / total_floating if total_floating > 0 else 0
    M = (1 - mine_density) / mine_density if mine_density > 0 else 0

    # -- Enumerate all valid bomb configurations for constrained cells --
    constrained_list = list(constrained_set)
    valid_configs = [] # Each entry: (assignment, bomb_count)
    n_constrained = len(constrained_list)

    def config_valid(assignment: List[bool]) -> bool:
        """
        For every revealed cell, check that the bomb count in its adjacent constrained
        cells equals: cell.adjacent_count - (# adjacent flagged cells).
        """
        # Build a quick lookup from constrained cell to bomb assignment.
        assignment_dict = {constrained_list[i]: assignment[i] for i in range(n_constrained)}
        for i in range(rows):
            for j in range(cols):
                if board[i][j].is_revealed:
                    # Count flags already placed among adjacent cells.
                    flagged_count = 0
                    adjacent = get_adjacent_positions(board, (i, j))
                    for r, c in adjacent:
                        if board[r][c].is_flagged:
                            flagged_count += 1
                    # Count bombs in adjacent cells that are in the constrained set.
                    bomb_count = 0
                    for r, c in adjacent:
                        if (r, c) in constrained_set and assignment_dict.get((r, c), False):
                            bomb_count += 1
                    expected = board[i][j].adjacent_count - flagged_count
                    if bomb_count != expected:
                        return False
        return True
    
    # Enumerate over all 2^(#constrained) assignments.
    for config in range(2 ** n_constrained):
        assignment = [(config >> i) & 1 == 1 for i in range(n_constrained)]
        if config_valid(assignment):
            bomb_count = sum(assignment)
            valid_configs.append((assignment, bomb_count))

    # If not configuration is valid, fall back to a simple model.
    if not valid_configs:
        prob_board =  [[None if board[i][j].is_revealed else mine_density 
                        for j in range(cols)] for i in range(rows)]
        return prob_board
    
    # Find maximum bomb count among all valid configurations.
    max_bomb_count = max([bomb_count for (_, bomb_count) in valid_configs])

    # Compute weight for each configuration.
    config_weights = []
    for assignment, bomb_count in valid_configs:
        # Weight = M^(max_bomb_count - bomb_count)
        weight = (M ** (max_bomb_count - bomb_count)) if M != 0 else 1
        config_weights.append(weight)
    total_weight = sum(config_weights)

    # For each cell in the constrained set, compute probability of being a bomb
    constrained_prob = {}
    for idx, cell in enumerate(constrained_list):
        weight_sum_bomb = 0
        for (assignment, _), w in zip(valid_configs, config_weights):
            if assignment[idx]:
                weight_sum_bomb += w
        prob = weight_sum_bomb / total_weight if total_weight > 0 else 0
        constrained_prob[cell] = prob

    # For each cell in the unconstrained set, use the global mine density.
    unconstrained_prob = {cell: mine_density for cell in unconstrained_set}

    # Build the final probability board.
    prob_board = [[None for j in range(cols)] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            cell: Cell = board[i][j]
            if cell.is_revealed:
                cell.probability = None
            elif (i, j) in constrained_prob:
                cell.probability = constrained_prob[(i, j)]
            else:
                cell.probability = unconstrained_prob[(i, j)]
            prob_board[i][j] = cell.probability

    return prob_board

def get_adjacent_positions(board: Board, pos: tuple) -> List[tuple]:
    """
    Returns a list of valid adjacent positions for a given cell.
    
    Hint: Check for boundary conditions and return a list of valid positions.
    """
    rows, cols = len(board), len(board[0])
    positions = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            new_row, new_col = pos[0] + i, pos[1] + j
            if 0 <= new_row < rows and 0 <= new_col < cols:
                positions.append((new_row, new_col))
    return positions

def compute_probability(state: GameState, pos: tuple) -> Optional[float]:
    """
    Computes the probability for a single cell to contain a mine.
    
    Hint: Analyze neighboring cells and return None if the cell is revealed.
    """
    row, col = pos
    board: Board = state.board
    cell: Cell = board[row][col]

    if cell.is_revealed:
        return None
    if cell.probability is not None:
        return cell.probability
    else:
        prob_board = calc_probabilities(state)
        return prob_board[row][col]