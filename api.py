import ast
import os
import re
from functools import lru_cache
from typing import Optional, Set, Tuple

from logic import UltimateTicTacToeGame

from google import genai


FAST_MODEL = os.getenv("UTTT_API_MODEL", "gemini-3-flash-preview")
MAX_OUTPUT_TOKENS = 12
API_KEY = "AIzaSyCFQZrPPLkxa6jeapVT6nLPdcTTpwkUvyk"


def _get_client():
    return genai.Client(api_key=API_KEY)


def _legal_global_moves(game: UltimateTicTacToeGame) -> Set[Tuple[int, int]]:
    return {
        game.get_global_position(bi, bj, r, c)
        for (bi, bj, r, c) in game.get_available_moves()
    }


def _build_prompt(game: UltimateTicTacToeGame, legal_moves) -> str:
    curr_board = game.curr_board if game.curr_board is not None else "any"
    return (
        "Ultimate Tic Tac Toe move request.\n"
        "Board values: 1=X(api), -1=O(human), 0=empty.\n"
        f"Board 9x9: {game.board_rep.tolist()}\n"
        f"Current target sub-board: {curr_board}\n"
        f"Legal global moves (row,col): {sorted(legal_moves)}\n"
        "Return exactly one tuple: (row, col). No more tokens."
    )


def _extract_move(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None

    match = re.search(r"\(?\s*([0-8])\s*,\s*([0-8])\s*\)?", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return None

    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
        and 0 <= value[0] <= 8
        and 0 <= value[1] <= 8
    ):
        return int(value[0]), int(value[1])

    return None


def _generate_response_text(prompt: str, model: str) -> str:
    client = _get_client()
    response = client.models.generate_content(model=model, contents=prompt)


    return (getattr(response, "text", None) or "").strip()


def get_ai_move(
    game: UltimateTicTacToeGame,
    max_tries: int = 5,
    model: str = FAST_MODEL,
) -> Optional[Tuple[int, int]]:
    legal_moves = _legal_global_moves(game)
    if not legal_moves:
        return None

    prompt = _build_prompt(game, legal_moves)

    for _ in range(max_tries):
        text = _generate_response_text(prompt, model=model)
        move = _extract_move(text)
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
