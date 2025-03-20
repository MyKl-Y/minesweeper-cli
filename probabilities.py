"""
Module: probabilities
Description: Computes and provides mine probability estimates for unrevealed cells.
"""

from typing import List, Optional
from ms_types import Board

def calc_probabilities(board: Board) -> List[List[Optional[float]]]:
    """
    Calculates the probability that each unrevealed cell contains a mine.
    
    Hint: Use heuristics based on adjacent revealed cells to calculate probability.
    """
    pass

def compute_probability(board: Board, pos: tuple) -> Optional[float]:
    """
    Computes the probability for a single cell to contain a mine.
    
    Hint: Analyze neighboring cells and return None if the cell is revealed.
    """
    pass
