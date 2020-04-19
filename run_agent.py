import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


def run_agent(num_episodes=1):
    env = UnityEnvironment(file_name="env/Reacher20.app")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    num_agents = len(env_info.agents)

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)
    agent.actor_local.load_state_dict(torch.load("model/checkpoint_actor.pth", map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load("model/checkpoint_critic.pth", map_location='cpu'))

    for i in range(num_episodes):
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            scores += env_info.rewards
            states = next_states
            if np.any(env_info.local_done):
                break
        print(f"{i + 1} episode, averaged score: {np.mean(scores)}")


if __name__ == '__main__':
    run_agent()
