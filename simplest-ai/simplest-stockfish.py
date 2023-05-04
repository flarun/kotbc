import torch
from simplest_train import (
    ChessModel,
)  # replace "chess_model" with the name of your script defining the model

model_path = "path/to/your/model.pth"
model = ChessModel()
model.load_state_dict(torch.load(model_path))
model.eval()

import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("path/to/stockfish")

import time


def play_game(model, engine):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Model makes a move
            x = np.array(board_to_array(board)).astype(np.float32)
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)  # Add batch dimension
            y = model(x)
            y = y.detach().numpy()
            legal_moves = list(board.legal_moves)
            legal_move_indices = [move_to_index(move) for move in legal_moves]
            y = y[0][legal_move_indices]
            move_index = np.argmax(y)
            move = legal_moves[move_index]
            board.push(move)
        else:
            # Stockfish makes a move
            result = engine.play(board, chess.engine.Limit(time=2.0))
            board.push(result.move)

    return board.result()


num_games = 10
results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

for i in range(num_games):
    result = play_game(model, engine)
    results[result] += 1
    print(f"Game {i+1}: {result}")

print(f"Results after {num_games} games: {results}")
