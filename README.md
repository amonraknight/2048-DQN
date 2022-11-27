# 2048-DQN：A DQN solution to game 2048 for learning purpose in Python.

## Technical Features
The prototype of the DQN including the neural network, brain, agent, memory are from [YangRui2015/2048_env](https://github.com/YangRui2015/2048_env). Thanks to this author for the sharing.  
* Convolutional neural network in Pytorch
* Double DQN
* Priority Experience Replay
* Epsilon Decay

The game logic and GUI are from [yangshun/2048-python](https://github.com/yangshun/2048-python). This small game inspired me to try to play with AI.
* GUI in tkinter

I added some new features on the basis of the 2 authors.
* Train on CPU/GPU.
* Modules from previous trainings can be loaded to continue the training. However, as the memory is not restorable, the continued training will start from a much less valuable group of memory.
* The reward is not only the score of an action. The changes in monotonicity is another parameter in reward calculation, as it is always prefered by human players that all the cells are always increasing or decreasing in one direction.
* Avoid the invalid actions. The network may ask the cells to move up when they can't. I gave the invalid action a negative reward. Moreover, when the brain gives invalid actions repetitively, choose a random step instead. 

## Library Versions
* Python 3.9
* Pytorch 1.0.2


## Usage
1. Training

  Go to game/game_nodisplay.py, execute _dqn_train()_ to start training. Modules indexed by episode will be generated and saved under outputs/.
2. Testing

  Please make sure that you have at least one module under outputs/. On game/game_nodisplay.py, comment _dqn_train()_ out and execute _dqn_solve()_.
3. Watch AI Play

  This is just for fun. Execute game/game_gird.py when you have got a wise module.
  
## Performance
I trained about 2 days in 70,000 episodes and got a network win 13 games output 1000 tests. A result exciting but not good enough.
