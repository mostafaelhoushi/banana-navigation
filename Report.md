# Report
[plot1]: ./plot.png "Trained PLot"

## Learning Algorithm
We train a neural network to deduce, for each possible state, the reward of each possible action.
To train that network, we first direct the agent to take random actions - no matter what the state is - to create a dataset showing the different rewards of various actions for each state.
In this exploration stage, this data is then used to train the neural network to map the input state to the Q function, i.e., the estimated reward for each possible action.
Then, we have an exploitation stage where the partially trained neural network guides us to choose the action with highest reward for a given input, while continuing to train the model.
We alternate between the exploration and exploitation to get the best possible model.

### Chosen Hyperparameters

### Model Architecture
A typical fully-connected neural network architecture consists of cascaded pairs of linear and non-linear layers. 
The size of the input linear layer in our case is the state size, and the size of the output linear layer is the number of possible actions.
Our `QNetwork` class defined in `model.py` allows for an arbitrary number of layers with arbitrary sizes to be defined by the user. 
For the example scripts and weights file provided, we used 2 hidden layers each with size of 64.

## Plot of Rewards
The following is the plot of reward at evert episode:
![Training Plot][plot1]

## Ideas for Future Work

- Increase number of training episodes
- Increase depth of network
