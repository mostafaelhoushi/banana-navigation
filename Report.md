# Report
[plot1]: ./plot.png "Trained PLot"

## Learning Algorithm
Reinforcement learning involves an agent living in an environment, and at each state of the environment, it takes an action that results in a reward as well as the change of the state. The aim of reinforcement learning is to maximize the accumulative reward.
Deep learning when used for supervised learning or classification, involves a neural network with input feature vector, and an output vector that shows the probability of each of the possible classes.
On the other hand, deep learning when used for reinforcement learning, involves a neural network with input state vector, and an output vector with the estimated accumulative reward for each possible action.

We train a neural network to deduce, for each possible state, the reward of each possible action.
To train that network, we first direct the agent to take random actions - no matter what the state is - to create a dataset showing the different rewards of various actions for each state.
In this exploration stage, this data is then used to train the neural network to map the input state to the Q function, i.e., the estimated reward for each possible action.
Then, we have an exploitation stage where the partially trained neural network guides us to choose the action with highest reward for a given input, while continuing to train the model.
We alternate between the exploration and exploitation to get the best possible model.

### Chosen Hyperparameters
- replay buffer size of 10K
- minibatch size of 64
- learning rate of 5 e-4
- discount factor, gamma of 0.99
- tau value for soft update of target parameters of 1e-3
- how often to update the network: every 4 steps
- epsilon initialized at 1, and decays with rate 0.995 till a minimum value of 0.01

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
- Use DDQN (Double Deep Q-Network)
- Use N-Step Q-Learning
- Use Prioritized Experience Replay
- Use Dueling Q-Network
- Use Distributional RL
- Use Noisy Network

Combine the above to build the Rainbow model.
