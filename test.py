from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from dqn import dqn

'''
change the file_name parameter to match the location of the Unity environment that you downloaded.

Mac: "path/to/Banana.app"
Windows (x86): "path/to/Banana_Windows_x86/Banana.exe"
Windows (x86_64): "path/to/Banana_Windows_x86_64/Banana.exe"
Linux (x86): "path/to/Banana_Linux/Banana.x86"
Linux (x86_64): "path/to/Banana_Linux/Banana.x86_64"
Linux (x86, headless): "path/to/Banana_Linux_NoVis/Banana.x86"
Linux (x86_64, headless): "path/to/Banana_Linux_NoVis/Banana.x86_64"
'''
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86")

brain_name = env.brain_names[0]                    # get the default brain
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
state_size = len(state)
action_size = brain.vector_action_space_size
agent = Agent(state_size=state_size,               # create agent
	action_size=action_size, seed=0) 
agent.load_weights('checkpoint.pth')               # load weights
score = 0                                          # initialize the score
eps = 0.1
while True:
    action = agent.act(state, eps)                 # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))

# close the environment
env.close()
