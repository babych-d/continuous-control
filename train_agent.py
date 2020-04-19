from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


def ddpg(n_episodes=250, max_t=1000, print_every=25):
    env = UnityEnvironment(file_name="env/Reacher20.app")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    num_agents = len(env_info.agents)

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for i in range(num_agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i], t)
            states = next_states
            score += rewards
            if np.any(dones):
                break

        scores_deque.append(score.mean())
        scores.append(score.mean())
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) > 30:
            print("Model trained successfully")
            torch.save(agent.actor_local.state_dict(), "model/checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "model/checkpoint_critic.pth")
            break

    return scores


def save_scores(scores):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("model/scores.png")


def main():
    scores = ddpg()
    save_scores(scores)


if __name__ == '__main__':
    main()
