## Continuous control project

This repository provides way to train Deep Reinforcement 
Learning agent in order to reach the target are with jointed arm 
in virtual environment.

In order to run this code, you'll need Unity Environment that 
Udacity team provided in Deep Reinforcement Learning Nanodegree. 

### Environment

In this environment, a double-jointed arm can move to target locations. 
The goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

Environment is considered solved when agent receives score 30 on average for 100 episodes.

### Training 

For training, see file `train_agent.py`

### Running

I committed one trained file in model/ folder that you can use 
to run this code without training. Just run:
```
python run_agent.py
```
