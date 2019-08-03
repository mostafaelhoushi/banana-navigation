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

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# train the agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)
scores = dqn(env, brain_name, agent)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# close the environment
env.close()
