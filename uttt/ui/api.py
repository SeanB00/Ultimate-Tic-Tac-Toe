import ast
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uttt.game.logic import UltimateTicTacToeGame

from google import genai


FAST_MODEL = "gemini-3-flash-preview"
MAX_OUTPUT_TOKENS = 12
API_KEY = "AIzaSyCFQZrPPLkxa6jeapVT6nLPdcTTpwkUvyk"

@lru_cache(maxsize=1)
def get_client():
    return genai.Client(api_key=API_KEY)


def legal_global_moves(game: UltimateTicTacToeGame) -> Set[Tuple[int, int]]:
    return {
        game.get_global_position(bi, bj, r, c)
        for (bi, bj, r, c) in game.get_available_moves()
    }


def build_prompt(game: UltimateTicTacToeGame, legal_moves) -> str:
    curr_board = game.curr_board if game.curr_board is not None else "any"
    return (
        "Ultimate Tic Tac Toe move request.\n"
        "Board values: 1=X(api), -1=O(human), 0=empty.\n"
        f"Board 9x9: {game.board_rep.tolist()}\n"
        f"Current target sub-board: {curr_board}\n"
        f"Legal global moves (row,col): {sorted(legal_moves)}\n"
        "Return exactly one tuple: (row, col). No more tokens."
    )


def extract_move(text: str):
    if not text:
        return None

    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return None
    return int(value[0]), int(value[1])



def generate_response_text(prompt: str, model: str) -> str:
    client = get_client()
    response = client.models.generate_content(model=model, contents=prompt)


    return (getattr(response, "text", None) or "").strip()


def get_ai_move(
    game: UltimateTicTacToeGame,
    max_tries: int = 5,
    model: str = FAST_MODEL,
) -> Optional[Tuple[int, int]]:
    legal_moves = legal_global_moves(game)
    if not legal_moves:
        return None

    prompt = build_prompt(game, legal_moves)

    for _ in range(max_tries):
        text = generate_response_text(prompt, model=model)
        move = extract_move(text)
        if move in legal_moves:
            return move
        prompt = (
            f"Invalid response: {text!r}. "
            f"Pick one legal move only: {sorted(legal_moves)}. "
            "Return exactly (row, col)."
        )

    return None


if __name__ == "__main__":
    game = UltimateTicTacToeGame()
    game.init_game()
    print(get_ai_move(game))
