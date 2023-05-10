# kotbc - king of the bell curve

## Simplest AI

This folder contains the simplest possible implementation of an Ai trained to output the most optimal move given as an input the board state.

### How to train the models

In order to train you will need a collection of .pgn files of relatively good chess games.

A recommended resource is the following: [https://database.nikonoel.fr/], Credits go to https://lichess.org/@/nikonoel.

Once you found your .pgn files, collect them into a folder and store it inside of 'simplest-ai/'.

The name of the folder is now the 'data_folder' variale you will specify later.

Navigate to the simplest-ai folder
'''
cd simplest-ai/
'''

To train, specify the arguments in the command line:
'''
python simplest_train.py <data_folder> <epochs> <batch_size> <learning_rate>
'''

The trained model will be stored at 'simplest-ai/models/', with the name in the following descriptive formatting: 'simplest-ai*DATA{data_folder}\_E{epochs}\_BS{batch_size}\_LR{learning_rate}*{now}.pth'
