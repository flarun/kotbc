import ast
from contextlib import nullcontext
from pathlib import Path

import chess
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_function(relative_path, function_name):
    source = (ROOT / relative_path).read_text()
    module = ast.parse(source)
    node = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    namespace = {"chess": chess, "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), namespace)
    return namespace[function_name]


def test_board_encoding_distinguishes_white_and_black_pieces():
    board_to_array = load_function("chess_ai/train.py", "board_to_array")
    board = chess.Board()
    array = board_to_array(board)

    # e1 is file 4, rank 0; e8 is file 4, rank 7.
    assert array[10, 4, 0] == -1.0
    assert array[11, 4, 0] == 1.0
    assert array[10, 4, 7] == 1.0
    assert array[11, 4, 7] == 1.0

    assert np.count_nonzero(array) == 64


class FakeMove:
    def __init__(self, name):
        self.name = name


class FakeBoard:
    def __init__(self):
        self.legal_moves = [FakeMove("a"), FakeMove("b")]
        self.current = None

    def push(self, move):
        self.current = move.name

    def pop(self):
        self.current = None


class FakeModel:
    def __call__(self, tensor):
        position = tensor[0]
        return FakeValue(0.9 if position == "a" else 0.1)


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTorch:
    no_grad = staticmethod(nullcontext)
    float32 = object()

    @staticmethod
    def tensor(value, dtype=None):
        return value


def test_black_move_choice_minimizes_white_perspective_score():
    choose_black_move = load_function("chess_ai/play.py", "choose_black_move")
    choose_black_move.__globals__["torch"] = FakeTorch
    choose_black_move.__globals__["board_to_array"] = lambda board: board.current

    move = choose_black_move(FakeBoard(), FakeModel())

    assert move.name == "b"
