import os
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Define the chess dataset class
class ChessDataset(Dataset):
    def __init__(self, data_folder):
        self.games = []
        for filename in os.listdir(data_folder):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r") as f:
                game = chess.pgn.read_game(f)
                result = game.headers["Result"]
                board = game.board()
                for move in game.mainline_moves():
                    self.games.append((board.copy(), result))
                    board.push(move)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        board, result = self.games[idx]
        x = np.array(board_to_array(board)).astype(np.float32)
        y = np.array(result_to_array(result)).astype(np.float32)
        return x, y


# Define the neural network
class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 12, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 64 * 12)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


# Define the training function
def train_model(model, dataset, epochs, batch_size, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Define the utility functions for converting chess data to arrays
def board_to_array(board):
    rows = []
    for i in range(8):
        row = []
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                row.append(piece.symbol().lower())
            else:
                row.append(".")
        rows.append(row)
    flat_rows = [item for sublist in rows for item in sublist]
    feature_planes = np.zeros((12, 8, 8), dtype=np.float32)
    for i, char in enumerate(flat_rows):
        if char == ".":
            continue
        sign = 1 if char.islower() else -1
        char = char.upper()
        piece_index = "PNBRQK".index(char)
        feature_planes[piece_index * 2, i // 8, i % 8] = sign
        feature_planes[piece_index * 2 + 1, i // 8, i % 8] = 1
    return feature_planes


def result_to_array(result):
    if result == "1/2-1/2":
        return [0.5]
    elif result == "1-0":
        return [1.0]
    elif result == "0-1":
        return [0.0]


if __name__ == "__main__":
    data_folder = "simplest-ai/database/"
    dataset = ChessDataset(data_folder)
    model = ChessModel()
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    train_model(model, dataset, epochs, batch_size, learning_rate)
    # After training the model
    model_path = "simplest-ai/models/simplest_elite_100_64_0-001.pth"
    torch.save(model.state_dict(), model_path)
