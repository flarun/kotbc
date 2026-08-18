import time
import chess
import chess.engine
import torch
import numpy as np
import os
from train import ChessModel, board_to_array

# --- SETUP ---
MODEL_FILENAME = "YOUR_NEW_MODEL_NAME.pth" # Update this after training!
model_path = os.path.join("models", MODEL_FILENAME)

STOCKFISH_PATH = "C:/path/to/stockfish.exe" # Update to your Stockfish .exe path!
# -------------

model = ChessModel()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def play_game(model, engine):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            best_move = None
            best_value = -float('inf')
            
            with torch.no_grad():
                for move in board.legal_moves:
                    board.push(move)
                    x = np.array(board_to_array(board)).astype(np.float32)
                    x = torch.from_numpy(x).unsqueeze(0)
                    value = model(x).item()
                    board.pop()
                    
                    if value > best_value:
                        best_move = move
                        best_value = value
                        
            board.push(best_move)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

    return board.result()

num_games = 10
results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

for i in range(num_games):
    result = play_game(model, engine)
    results[result] = results.get(result, 0) + 1
    print(f"Game {i+1}: {result}")

print(f"Results after {num_games} games: {results}")

engine.quit()