import chess
import torch
from simplest_train import ChessModel, board_to_array

# Load the saved model
model_path = "simplest-ai/models/simplest_elite_100_64_0-001.pth"
model = ChessModel()
model.load_state_dict(torch.load(model_path))

# Create the chess board
board = chess.Board()

# Main game loop
while not board.is_game_over():
    print(board)
    if board.turn == chess.WHITE:
        # Ask user for move
        move_str = input("Enter your move in algebraic notation: ")
        move = chess.Move.from_uci(move_str)
        board.push(move)
    else:
        # Get predicted value for current position
        x = torch.tensor([board_to_array(board)])
        value = model(x).item()

        # Find the best move based on predicted value
        best_move = None
        best_value = -1
        for move in board.legal_moves:
            board.push(move)
            x = torch.tensor([board_to_array(board)])
            value = model(x).item()
            board.pop()
            if value > best_value:
                best_move = move
                best_value = value

        # Play the best move
        board.push(best_move)

# Print the final result
result = board.result()
print(f"Game over: {result}")
