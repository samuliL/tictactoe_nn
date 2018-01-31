# tictactoe_nn

## Introduction

The purpose of this project was to understand reinforcement learning in a simplistic setting. To achieve this, the game of tic-tac-toe was implemented with the possibility of human player, random AI, minmax (optimal, but slow) AI and a neural network-based AI. The random and minmax players were used to train the neural network. 

The training of the network is essentially an application of supervised learning. The network is shown all the moves from a batch of games, and each move is labeled either "good" or "bad" depending on what was the outcome of the corresponding game for the player making the move (usually "good" for win or draw and "bad" for loss). We consider the log propability of the move that was made as the loss function, and compute the gradient using backpropagation. If the move is labeled "good", we multiply the gradient with a positive coefficient and with a negative coefficient in case the move was "bad". This way, after running thousands of games the good moves tend to get encouraged and bad moves discouraged. This idea is mainly taken from [1], where it was used to train a neural network to learn to play the game pong.

The implementation allows for a flexible design of the neural network. Any number of hidden layers and neurons can be used, although we did not observe any major difference between using two hidden layer and 20 neurons, and two hidden layers with 60 neurons in each layer. The activations in the output layer are given by the softmax function, and for the hidden layers one can choose either rectified linear units (ReLU's) or sigmoids. In practice we observed sigmoids to be more stable numerically.

## Results

In the experiments we only considered a 3x3 board, but the implementation readily allows for larger boards. After some simple training the neural network is able to win the random player in about 85% of the games when playing with X's and about 60% of the games when playing with O's. This is significantly better than just making random moves, as for the minmax AI the corresponding numbers are around 95% and 80% (with the remaining 20% ending in draws). 

A more sophisticated training could probably improve the results. It would be interesting to see if the neural network can be used to train a clever AI for a, say, 10x10 board with 5 in-a-row required for winning, but this would probably require quite a lot of computation.

## References

[1] Andrej Karpathy's blog "Deep Reinforcement Learning: Pong from Pixels", http://karpathy.github.io/2016/05/31/rl/ 
