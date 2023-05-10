# kotbc - king of the bell curve

## Simplest AI

This folder contains the simplest possible implementation of an Ai trained to output the most optimal move given as an input the board state.

## Preliminar setup

This project uses conda with Python 3.11.3.

If you want to install the PyTorch CUDA drivers for faster workflow, refer to this documentation: [https://pytorch.org/get-started/locally/]

To create, activate and setup the needed environment:

```sh
conda create -n kotbc python=3.11.3
conda activate kotbc
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

### How to train the models

In order to train you will need a collection of .pgn files of relatively good chess games.

A recommended resource is the following: [https://database.nikonoel.fr/], Credits go to https://lichess.org/@/nikonoel.

Once you found your .pgn files, collect them into a folder and store it inside of `simplest-ai/`.

The name of the folder is now the `data_folder` variale you will specify later.

Navigate to the simplest-ai folder

```sh
cd simplest-ai/
```

To train, specify the arguments in the command line:

```sh
python3 simplest_train.py <data_folder> <epochs> <batch_size> <learning_rate>
```

A good set of values to start with is the following:

```sh
python3 simplest_train.py <data_folder> 200 64 0.001
```

The trained model will be stored at `simplest-ai/models/`, with the name in the following descriptive formatting: `simplest-ai*DATA{data_folder}\_E{epochs}\_BS{batch_size}\_LR{learning_rate}*{now}.pth`

### How to play against your models

TODO: UI for the chess match between user and AI

### How to evaluate your models against Stockfish

TODO: CSV file writing of results.
