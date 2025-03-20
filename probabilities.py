"""
Module: probabilities
Description: Computes and provides mine probability estimates for unrevealed cells.
"""

import time
import numpy as np
from typing import List, Optional, Tuple
from ms_types import Board, GameState, Cell

def calc_probabilities(state: GameState, sample_threshold: int = 2**15, sample_size: int = 10000) -> List[List[Optional[float]]]:
    """
    Calculates the probability that each unrevealed cell contains a mine.
    
    Hint: Use heuristics based on adjacent revealed cells to calculate probability.
    """
    ''' Original Slow Method
    board: Board = state.board
    rows, cols = len(board), len(board[0])

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

    valid_configs = [] # Each entry: (assignment, bomb_count)

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
    '''
    ''' Possible Faster Method 1
    board: Board = state.board
    rows, cols = len(board), len(board[0])

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
    unconstrained_set = {(i, j) for i in range(rows) for j in range(cols)
                        if not board[i][j].is_revealed and (i, j) not in constrained_set}

    # Total floating tiles = all unrevealed cells
    total_floating = len(constrained_set) + len(unconstrained_set)
    num_remaining_mines = state.num_mines - state.num_flags
    mine_density = num_remaining_mines / total_floating if total_floating > 0 else 0
    M = (1 - mine_density) / mine_density if mine_density > 0 else 0

    # -- Enumerate all valid bomb configurations for constrained cells --
    constrained_list = list(constrained_set)
    n_constrained = len(constrained_list)

    # Early Exit: if no constrained cells, assign global mine density to all unrevealed cells.
    if n_constrained == 0:
        prob_board = [[None if board[i][j].is_revealed else mine_density 
                        for j in range(cols)] for i in range(rows)]
        return prob_board

    # Generate all possible configurations for the constrained cells.
    # For n_constrained <= 32, we can use np.uint32; adjust if needed.
    num_configs = 2 ** n_constrained
    # Create an array of all integers from 0 to num_configs-1.
    ints = np.arange(num_configs, dtype=np.uint32)
    # View these integers as bytes, then unpack bits.
    # We need 32 bits per integer, then take the last n_constrained columns.
    bits = np.unpackbits(ints.view(np.uint8)).reshape(-1, 4 * 8)
    configs = bits[:, -n_constrained:].astype(bool)  # shape: (num_configs, n_constrained)

    # Precompute the list of revealed cells with their adjacent-constrained mask and expected bomb count.
    revealed_masks = []
    expected_counts = []
    for i in range(rows):
        for j in range(cols):
            if board[i][j].is_revealed:
                adj = get_adjacent_positions(board, (i, j))
                # Count flagged neighbors.
                flagged_count = sum(1 for r, c in adj if board[r][c].is_flagged)
                expected = board[i][j].adjacent_count - flagged_count
                # Create a boolean mask for constrained cells adjacent to (i, j).
                mask = np.array([((i2, j2) in adj) for (i2, j2) in constrained_list], dtype=bool)
                # Only consider revealed cells that have adjacent constrained cells.
                if mask.any():
                    revealed_masks.append(mask)
                    expected_counts.append(expected)
    
    # If there are no revealed constraints affecting constrained cells, fall back to global mine density.
    if not revealed_masks:
        prob_board = [[None if board[i][j].is_revealed else mine_density for j in range(cols)]
                        for i in range(rows)]
        return prob_board

    revealed_masks = np.array(revealed_masks)  # shape: (num_revealed, n_constrained)
    expected_counts = np.array(expected_counts)  # shape: (num_revealed,)

    # For each revealed cell, compute bomb counts for all configurations.
    # This yields an array of shape (num_configs, num_revealed)
    config_counts = configs @ revealed_masks.T  # dot product over the constrained cells axis

    # Check for validity: for each revealed cell, the bomb count must equal the expected count.
    valid_mask = np.all(config_counts == expected_counts, axis=1)  # shape: (num_configs,)
    valid_configs = configs[valid_mask]
    
    if valid_configs.shape[0] == 0:
        # If no valid configuration, use mine_density as fallback.
        prob_board = [[None if board[i][j].is_revealed else mine_density for j in range(cols)]
                        for i in range(rows)]
        return prob_board

    # Compute bomb count for each valid configuration.
    bomb_counts = valid_configs.sum(axis=1)  # shape: (num_valid_configs,)
    max_bomb_count = bomb_counts.max()

    # Compute weight for each valid configuration.
    weights = np.power(M, max_bomb_count - bomb_counts, dtype=float) if M != 0 else np.ones_like(bomb_counts, dtype=float)
    total_weight = weights.sum()

    # Compute probability for each constrained cell.
    # For each cell, get the weighted fraction of configurations where that cell is a bomb.
    constrained_prob = {}
    # valid_configs: shape (num_valid_configs, n_constrained)
    for idx, cell in enumerate(constrained_list):
        cell_weights = weights[valid_configs[:, idx]]
        constrained_prob[cell] = cell_weights.sum() / total_weight

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
    '''
    ''' Possible Faster Method 2
    board: Board = state.board
    rows, cols = len(board), len(board[0])
    
    # Separate unrevealed cells into two sets:
    # constrained_set: cells adjacent to at least one revealed cell.
    # unconstrained_set: cells not adjacent to any revealed cell.
    constrained_set = set()
    unconstrained_set = set()
    
    for i in range(rows):
        for j in range(cols):
            if not board[i][j].is_revealed:
                adjacent = get_adjacent_positions(board, (i, j))
                if any(board[r][c].is_revealed for r, c in adjacent):
                    constrained_set.add((i, j))
                else:
                    unconstrained_set.add((i, j))
    
    total_unrevealed = len(constrained_set) + len(unconstrained_set)
    num_remaining_mines = state.num_mines - state.num_flags
    mine_density = num_remaining_mines / total_unrevealed if total_unrevealed > 0 else 0
    M = (1 - mine_density) / mine_density if mine_density > 0 else 0

    # Create a list and index lookup for constrained cells.
    constrained_list = list(constrained_set)
    constrained_index = {cell: idx for idx, cell in enumerate(constrained_list)}
    N = len(constrained_list)
    
    # If there are no constrained cells, assign global mine density to all unrevealed cells.
    if N == 0:
        return [
            [None if board[i][j].is_revealed else mine_density for j in range(cols)]
            for i in range(rows)
        ]
    
    # Generate all possible configurations (2^N rows, each with N binary values).
    all_configs = np.arange(2 ** N, dtype=np.uint32)[:, None]
    shifts = np.arange(N, dtype=np.uint32)
    configs = ((all_configs >> shifts) & 1).astype(np.int8)  # Shape: (2^N, N)
    
    # Initialize valid configuration mask.
    valid_mask = np.ones(configs.shape[0], dtype=bool)
    
    # For each revealed cell that touches constrained cells, create a binary mask and enforce its constraint.
    for i in range(rows):
        for j in range(cols):
            if board[i][j].is_revealed:
                mask = np.zeros(N, dtype=np.int8)
                adjacent = get_adjacent_positions(board, (i, j))
                flagged_count = 0
                for r, c in adjacent:
                    if board[r][c].is_flagged:
                        flagged_count += 1
                    if (r, c) in constrained_set:
                        idx = constrained_index[(r, c)]
                        mask[idx] = 1
                # Only apply if the revealed cell touches any constrained cell.
                if mask.sum() > 0:
                    expected = board[i][j].adjacent_count - flagged_count
                    counts = configs.dot(mask)
                    valid_mask &= (counts == expected)
                    if not valid_mask.any():
                        break  # No valid configuration remains.
        else:
            continue
        break  # Early exit if a break occurred in the inner loop.
    
    valid_configs = configs[valid_mask]
    
    # If no valid configuration is found, fallback to global mine density.
    if valid_configs.size == 0:
        return [
            [None if board[i][j].is_revealed else mine_density for j in range(cols)]
            for i in range(rows)
        ]
    
    # Compute the number of bombs in each valid configuration.
    bomb_counts = valid_configs.sum(axis=1)
    max_bomb_count = bomb_counts.max()
    
    # Compute configuration weights using vectorized power.
    # If M is zero (global density is zero), simply use weight 1.
    if M == 0:
        weights = np.ones(valid_configs.shape[0], dtype=float)
    else:
        weights = np.power(M, (max_bomb_count - bomb_counts))
    total_weight = weights.sum()
    
    # For each constrained cell, compute its probability as the weighted sum of 
    # configurations in which that cell is a bomb.
    weighted_bomb_sum = (weights[:, None] * valid_configs).sum(axis=0)
    cell_probs = weighted_bomb_sum / total_weight
    
    # Build the final probability board.
    prob_board = []
    for i in range(rows):
        for j in range(cols):
            cell: Cell = board[i][j]
            if cell.is_revealed:
                cell.probability = None
            elif (i, j) in constrained_set:
                cell.probability = cell_probs[constrained_index[(i, j)]]
            else:
                cell.probability = mine_density
            prob_board.append(cell.probability)

    return prob_board
    '''
    ''' Possible Faster Method 3'''
    board: Board = state.board
    rows, cols = len(board), len(board[0])
    
    # Separate unrevealed cells into two sets:
    # constrained_set: cells adjacent to at least one revealed cell.
    # unconstrained_set: cells not adjacent to any revealed cell.
    constrained_set = set()
    unconstrained_set = set()
    
    for i in range(rows):
        for j in range(cols):
            if not board[i][j].is_revealed:
                adjacent = get_adjacent_positions(board, (i, j))
                if any(board[r][c].is_revealed for r, c in adjacent):
                    constrained_set.add((i, j))
                else:
                    unconstrained_set.add((i, j))
    
    total_unrevealed = len(constrained_set) + len(unconstrained_set)
    num_remaining_mines = state.num_mines - state.num_flags
    mine_density = num_remaining_mines / total_unrevealed if total_unrevealed > 0 else 0
    M = (1 - mine_density) / mine_density if mine_density > 0 else 0

    # Create a list and index lookup for constrained cells.
    constrained_list = list(constrained_set)
    constrained_index = {cell: idx for idx, cell in enumerate(constrained_list)}
    N = len(constrained_list)
    
    ## Precompute masks and expected bomb counts for revealed cells touching constrained cells.
    revealed_masks = []
    revealed_targets = []
    for i in range(rows):
        for j in range(cols):
            if board[i][j].is_revealed:
                mask = np.zeros(N, dtype=np.int8)
                adjacent = get_adjacent_positions(board, (i, j))
                flagged_count = 0
                for r, c in adjacent:
                    if board[r][c].is_flagged:
                        flagged_count += 1
                    if (r, c) in constrained_set:
                        idx = constrained_index[(r, c)]
                        mask[idx] = 1
                if mask.sum() > 0:
                    revealed_masks.append(mask)
                    revealed_targets.append(board[i][j].adjacent_count - flagged_count)
    
    # Decide whether to use full enumeration or Monte Carlo sampling.
    total_configurations = 2 ** N
    if total_configurations <= sample_threshold:
        # Full enumeration via vectorized bit manipulation.
        all_configs = np.arange(total_configurations, dtype=np.uint32)[:, None]
        shifts = np.arange(N, dtype=np.uint32)
        configs = ((all_configs >> shifts) & 1).astype(np.int8)  # Shape: (2^N, N)
        
        valid_mask = np.ones(configs.shape[0], dtype=bool)
        for mask, target in zip(revealed_masks, revealed_targets):
            counts = configs.dot(mask)
            valid_mask &= (counts == target)
            if not valid_mask.any():
                break
        valid_configs = configs[valid_mask]
    else:
        # Use Monte Carlo sampling to approximate the valid configurations.
        valid_configs_list = []
        attempts = 0
        max_attempts = sample_size * 10  # Limit to avoid infinite loops.
        while len(valid_configs_list) < sample_size and attempts < max_attempts:
            assignment = np.random.randint(0, 2, size=(N,), dtype=np.int8)
            valid = True
            for mask, target in zip(revealed_masks, revealed_targets):
                if (assignment * mask).sum() != target:
                    valid = False
                    break
            if valid:
                valid_configs_list.append(assignment)
            attempts += 1
        if len(valid_configs_list) == 0:
            # Fallback: if no valid configuration is found, use global mine density.
            return [
                [None if board[i][j].is_revealed else mine_density for j in range(cols)]
                for i in range(rows)
            ]
        valid_configs = np.array(valid_configs_list)
    
    # Compute the number of bombs in each valid configuration.
    bomb_counts = valid_configs.sum(axis=1)
    max_bomb_count = bomb_counts.max()
    
    # Compute configuration weights using vectorized power.
    # If M is zero (global density is zero), simply use weight 1.
    if M == 0:
        weights = np.ones(valid_configs.shape[0], dtype=float)
    else:
        weights = np.power(M, (max_bomb_count - bomb_counts))
    total_weight = weights.sum()
    
    # For each constrained cell, compute its probability as the weighted sum of 
    # configurations in which that cell is a bomb.
    weighted_bomb_sum = (weights[:, None] * valid_configs).sum(axis=0)
    cell_probs = weighted_bomb_sum / total_weight
    
    # Build the final probability board.
    prob_board = []
    for i in range(rows):
        for j in range(cols):
            cell: Cell = board[i][j]
            if cell.is_revealed:
                cell.probability = None
            elif (i, j) in constrained_set:
                cell.probability = cell_probs[constrained_index[(i, j)]]
            else:
                cell.probability = mine_density
            prob_board.append(cell.probability)

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