[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[plot1]: ./plot.png "Trained PLot"

# Banana Navigation using Deep Reinforcement Learning

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world using deep reinforcement learning.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Clone the repo:
```
git clone https://github.com/mostafaelhoushi/banana-navigation.git
```

2. Change directory into the repo:
```
cd banana-navigation
```

3. Download the Unity environment using one of the commands below.  You need only select the environment that matches your operating system:
- Linux: 
```
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
```
- Mac OSX:
```
curl -O https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
```
- Windows (32-bit): [PowerShell]
```
$client = new-object System.Net.WebClient
$client.DownloadFile("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip")
```
- Windows (64-bit):
```
$client = new-object System.Net.WebClient
$client.DownloadFile("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip")
```
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

4. Unzip (or decompress) the downloaded file: 
- Linux: 
```
unzip Banana_Linux.zip
```
- Mac OSX:

```
curl -O https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
```
- Windows (32-bit): [PowerShell]
```
Expand-Archive -Path Banana_Windows_x86.zip -DestinationPath .
```
- Windows (64-bit): [PowerShell]
```
Expand-Archive -Path Banana_Windows_x86_64.zip -DestinationPath .
```

5. Create (and activate) a new environment with Python 3.6.
- Linux or Mac OSX:
```
conda create --name drlnd python=3.6
source activate drlnd
```
- Windows:
```
conda create --name drlnd python=3.6
activate drlnd
```

6. Install the OpenAI gym library and PyTorch libraries:
```
pip install gym torch torchvision
```

7. Edit the `train.py` and `test.py` scripts by updating the `file_name` parameter in the first function call in the file. Read the comments in the file to determine what path you should use.

8. Run the training script:
```
python train.py
```

You should expect an output similar to this:
```
Found path: /home/melhoushi/Udacity/banana-navigation/./Banana_Linux/Banana.x86_64
Mono path[0] = '/home/melhoushi/Udacity/banana-navigation/./Banana_Linux/Banana_Data/Managed'
Mono config path = '/home/melhoushi/Udacity/banana-navigation/./Banana_Linux/Banana_Data/MonoBleedingEdge/etc'
Preloaded 'ScreenSelector.so'
Preloaded 'libgrpc_csharp_ext.x64.so'
Unable to preload the following plugins:
	ScreenSelector.so
	libgrpc_csharp_ext.x86.so
Logging to /home/melhoushi/.config/unity3d/Unity Technologies/Unity Environment/Player.log
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
Number of agents: 1
Number of actions: 4
States look like: [1.         0.         0.         0.         0.84408134 0.
 0.         1.         0.         0.0748472  0.         1.
 0.         0.         0.25755    1.         0.         0.
 0.         0.74177343 0.         1.         0.         0.
 0.25854847 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345 0.
 0.        ]
States have length: 37
Episode 100	Average Score: 0.976
Episode 200	Average Score: 4.67
Episode 300	Average Score: 8.34
Episode 400	Average Score: 10.34
Episode 500	Average Score: 12.89
Episode 503	Average Score: 13.03
Environment solved in 403 episodes!	Average Score: 13.03
```

and finally you should find the following plot:
![Training Plot][plot1]

9. Click on the "X" button to close the plotting window.

10. You should now find a `checkpoint.pth` file in your directory that contains the PyTorch model of your training agent.

11. You can now run the testing script:
```
python test.py
```


### Notebook
You may also explore and run the `Navigation.ipynb` notebook.

1. Create an IPython kernel for the `drlnd` environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

2. Start Jupyter Notebook:
```
jupyter notebook
```

3. Copy the URL message that you see and paste it into your browser.

4. In the browser, open the `Navigation.ipynb` notebook.

5. Before running code in a notebook, change the kernel to match the drlnd environment by clicking on the top menu: Kernel -> Change Kernel -> drlnd
