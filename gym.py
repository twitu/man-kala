# Author: Till Zemann
# License: MIT License

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

from game import GameSimulator, MinimaxPlayer, RandomPlayer
from mancala_agent import MancalaAgent
import game

# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.

# env = gym.make("Blackjack-v1", sab=True)


done = False
# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = MancalaAgent(
    None,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

p1 = RandomPlayer(game.RANDOM_SEED)
game.FIRST_PLAYER_AGENT = True
env: GameSimulator = GameSimulator([agent, p1])
agent.env = env
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # Play opponent turn if agent is not the first player
    if (game.FIRST_PLAYER_AGENT and env.turn == 1) or (
        not game.FIRST_PLAYER_AGENT and env.turn == 0
    ):
        env.play_next_turn()
        obs = env.observation

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # simulate opponent turn
        env.play_next_turn()
        next_obs = env.observation

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
