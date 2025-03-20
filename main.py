"""
Module: main
Description: The main entry point for the Minesweeper CLI application.
"""
import sys
from ms_types import GameState, Mode
from board import init_board, place_mines, update_board, place_mines_after_first_move
from input import get_action_from_input, set_raw_mode, restore_terminal, Action
from render import render_state
from solver import apply_solver, check_win_condition
from probabilities import calc_probabilities
# You may also import update_state from a game_logic module if you choose to separate it.

def new_game(rows: int, cols: int, difficulty: str, method: str) -> GameState:
    """
    Initializes a new game state given board dimensions, difficulty level, and solver method.
    
    This function creates a new board, determines the number of mines based on the difficulty,
    places the mines (ensuring the first move is safe), and reveals the starting cell.
    """
    board = init_board(rows, cols)
    if difficulty == "easy":
        num_mines = int(rows * cols * 0.12)
    elif difficulty == "medium":
        num_mines = int(rows * cols * 0.16)
    elif difficulty == "hard":
        num_mines = int(rows * cols * 0.20)
    else:
        num_mines = int(rows * cols * 0.16)  # default to medium
    
    board = place_mines_after_first_move(board, num_mines, (0, 0))
    board = update_board(board, (0, 0))
    return GameState(
        board=board,
        cursor_pos=(0, 0),
        game_mode=Mode.REVEAL,
        game_over=False,
        win=False,
        num_mines=num_mines,
        num_flags=0,
        exit=False
    )

def update_state(action, state: GameState, num_mines: int, method: str, difficulty: str) -> GameState:
    """
    Updates the game state according to a user action.
    
    Hint: Handle actions like moving the cursor, revealing cells, toggling modes, etc.
    """
    current_row, current_col = state.cursor_pos
    rows, cols = state.board.shape

    if action == Action.MOVE_UP:
        new_row = max(0, current_row - 1)
        state.cursor_pos = (new_row, current_col)
    elif action == Action.MOVE_DOWN:
        new_row = min(rows - 1, current_row + 1)
        state.cursor_pos = (new_row, current_col)
    elif action == Action.MOVE_LEFT:
        new_col = max(0, current_col - 1)
        state.cursor_pos = (current_row, new_col)
    elif action == Action.MOVE_RIGHT:
        new_col = min(cols - 1, current_col + 1)
        state.cursor_pos = (current_row, new_col)
    elif action == Action.TOGGLE_FLAG:
        state.game_mode = Mode.FLAG
        state.board[current_row][current_col].is_flagged = not state.board[current_row][current_col].is_flagged
        if state.board[current_row][current_col].is_flagged:
            state.num_flags += 1
        else:
            state.num_flags -= 1
    elif action == Action.REVEAL_CELL:
        state.game_mode = Mode.REVEAL
        state.board = update_board(state.board, state.cursor_pos)
        if state.board[state.cursor_pos[0]][state.cursor_pos[1]].has_mine:
            state.game_over = True
    elif action == Action.TOGGLE_SOLVER:
        state.game_mode = Mode.SOLVER
        state = apply_solver(state, method, num_mines)
    elif action == Action.TOGGLE_PROBABILITY:
        state.game_mode = Mode.PROB
        calc_probabilities(state)
    elif action == Action.QUIT:
        state.exit = True
    elif action == Action.NEW_GAME:
        state = new_game(rows, cols, difficulty, method)

    return state

def game_loop(state: GameState, num_mines: int, method: str, difficulty: str):
    """
    Main game loop: reads input, updates state, and re-renders until the game is over.
    """
    while not state.exit:
        if check_win_condition(state.board, num_mines):
            state.win = True
            state = apply_solver(state, method, num_mines)
        render_state(state)
        action = get_action_from_input()
        state = update_state(action, state, num_mines, method, difficulty)

def main(rows: int, cols: int, difficulty: str, method: str):
    """
    Initializes the game state, sets up the terminal, and starts the game loop.
    """
    set_raw_mode()
    try:
        board = init_board(rows, cols)
        #board = place_mines(board, mines)
        #first_non_mine = (0, 0)
        #while board[first_non_mine[0]][first_non_mine[1]].has_mine:
        #    board = place_mines(board, mines)
        #board = update_board(board, first_non_mine)
        #state = GameState(board=board, cursor_pos=first_non_mine, game_mode=Mode.REVEAL, game_over=False)

        num_mines = 0

        if difficulty == "easy":
            num_mines = int(rows * cols * .12 // 1)
        elif difficulty == "medium":
            num_mines = int(rows * cols * .16 // 1)
        elif difficulty == "hard":
            num_mines = int(rows * cols * .20 // 1)

        board = place_mines_after_first_move(board, num_mines, (0, 0))
        update_board(board, (0, 0))
        state = GameState(
            board=board, 
            cursor_pos=(0,0), 
            game_mode=Mode.REVEAL, 
            game_over=False, 
            win=False, 
            num_mines=num_mines, 
            num_flags=0, 
            exit=False
        )
        game_loop(state, num_mines, method, difficulty)
    finally:
        restore_terminal()

if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    difficulty = sys.argv[3] if len(sys.argv) > 3 and [sys.argv[3]] == [val for val in ["easy", "medium", "hard"] if val == sys.argv[3]] else "medium"
    solver_method = sys.argv[4] if len(sys.argv) > 4 else "none"

    main(rows, cols, difficulty, solver_method)
