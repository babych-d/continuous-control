##Report

This implementation uses Actor-Critic method.

Actor model consists of 2 fully-connected layers with tanh activation on top

Critic includes 4 fully-connected layers where action is included in 2nd one.
No activation on top.

Agent uses following hyperparameters: 
```
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay
```

Agent trains 10 times every 20 timestamps

Plot with rewards can be found in model/scores.png
