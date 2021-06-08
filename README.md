# Project Continuous Control

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

There are 2 agents. Each observes a state with length: 24, corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### Unity Environment
After downloading the Unity Environment and saving it into the same folder where the main notebook `Continuous_Control.ipynb` sits, one can load the env by running the following code:

> `env = UnityEnvironment(file_name='Tennis.app')`
>
>  `brain_name = env.brain_names[0]`
>
>  `brain = env.brains[brain_name]`
>
>  `env_info = env.reset(train_mode=True)[brain_name]`


### Problem Definition and success criteria
In RL, any problem should be defined as MDP (Markov Decision Process) that consists of States, Actions, Reward function and Transition Function

#### States

The state space consists of 24 variables corresponding to the position and velocity of the ball and racket.
Given this information, the agent has to learn how to best select actions.  

One can check how the state vector looks like for one agent by running the following code:

> `state = env_info.vector_observations[0]`
> `print(state)`


#### Actions
Each action is a vector with 2 numbers, corresponding to corresponding to movement toward (or away from) the net, and jumping.
Every entry in the action vector must be a number between -1 and 1. This is important, so we make sure in the code to clip actions to get to this.

One can check the number of actions by:

> `print(brain.vector_action_space_size)`


#### Rewards

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.


#### When the task is considered solved

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### Getting Started

1. Clone the repository.
   
   git clone https://github.com/alinapopaqb/drl-collaboration-competition.git
 
2. To set up your python environment to run the code in this repository, follow the instructions below.
      
      Create (and activate) a new environment with Python 3.6.
      
      Linux or Mac:
      
      `conda create --name drlnd python=3.6`
      
      `source activate drlnd`
      
      Windows:
      
      `conda create --name drlnd python=3.6 `
      
      `activate drlnd`
      
3. Install the dependencies in the newly created env by running in the shell:
     
     `pip install -r requirements.txt`

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
       - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
       - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
       - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
       - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
       
5. Place the file in the `drl-collaboration-competition/` folder, and unzip (or decompress) the file. 

6. Create an IPython kernel for the drlnd environment:

    `python -m ipykernel install --user --name drlnd --display-name "drlnd"`

7. Run in the terminal of the current  directory:

    `jupyter notebook`

8. Go to the `Tennis.ipynb` and run the code there based on the instructions.
