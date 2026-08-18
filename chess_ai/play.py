import chess
import torch
import os
from train import ChessModel, board_to_array

# --- SETUP ---
MODEL_FILENAME = "YOUR_NEW_MODEL_NAME.pth" # Update this after training!
model_path = os.path.join("models", MODEL_FILENAME)

model = ChessModel()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
# -------------

board = chess.Board()

while not board.is_game_over():
    print(board)
    print("-" * 20)
    
    if board.turn == chess.WHITE:
        move_str = input("Enter your move in algebraic notation (e.g., e4, Nf3) or UCI (e.g., e2e4): ")
        try:
            move = board.parse_san(move_str)
        except ValueError:
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print("Invalid move. Try again.")
                continue
        board.push(move)
    else:
        best_move = None
        best_value = -float('inf')
        
        with torch.no_grad():
            for move in board.legal_moves:
                board.push(move)
                x = torch.tensor([board_to_array(board)], dtype=torch.float32)
                value = model(x).item()
                board.pop()
                
                if value > best_value:
                    best_move = move
                    best_value = value

        print(f"AI plays: {best_move}")
        board.push(best_move)

print(f"Game over: {board.result()}")