from collections import deque, namedtuple
import os
import random, math
import time
from game import GameSimulator, PER_PLAYER_TILE, OBSERVATION_LEN


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TAU = 0.005
LR = 1e-4

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class RlMancalaPlayer:
    def __init__(self, env: GameSimulator = None):
        # Get number of actions from gym action space
        self.n_actions = PER_PLAYER_TILE
        # Get the number of state observations
        self.n_observations = OBSERVATION_LEN
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.env = env
        self.name = "Pr00Play3r"
        self.full_name = "RlMancalaPlayer"

    def save_model(self, path="model_weights.pth"):
        timestamp = time.strftime("%Y%m%d%H%M%S")
        torch.save(self.policy_net.state_dict(), f"{timestamp}_policy_{path}")
        torch.save(self.target_net.state_dict(), f"{timestamp}_target_{path}")

    def load_model(self, policy_path, target_path):
        
        # # If path is not a file, assume it's a directory and look for the latest model_weights file
        # if not os.path.isfile(path):
        #     files = os.listdir(path)
        #     files = [f for f in files if f.startswith('20') and f.endswith('.pth')]
        #     files.sort()
        #     if len(files) == 0:
        #         raise ValueError("No saved model found")
        #     path = os.path.join(path, files[-1])

        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.target_net.load_state_dict(torch.load(target_path))
        self.policy_net.eval()
        self.target_net.eval()
        self.policy_net.to(device)
        self.target_net.to(device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1

        valid_moves = self.env.action_space
        if len(valid_moves) == 0:
            return torch.tensor([[0]], device=device, dtype=torch.long)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                index = self.policy_net(torch.tensor(state, device=device, dtype=torch.float)).max(0).indices.view(1, 1).long()
                if not index in valid_moves:
                    return torch.tensor(
                        [[random.choice(valid_moves)]],
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    return index
        else:
            return torch.tensor(
                [[random.choice(self.env.action_space)]],
                device=device,
                dtype=torch.long,
            )

    def play_turn(self, sim: GameSimulator) -> int:
        valid_moves = self.env.action_space
        if len(valid_moves) == 0:
            return 0
        elif len(valid_moves) == 1:
            return valid_moves[0]
        else:
            index = self.policy_net(torch.tensor(sim.observation, device=device, dtype=torch.float)).max(0).indices.view(1, 1).long()
            # hack because network doesn't understand illegal moves
            if not index in valid_moves:
                return random.choice(valid_moves)
            else:
                return index

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # print("optimize model....")
        # print(f"{batch.state}")
        state_batch = torch.stack(batch.state, dim=1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch.T).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states.reshape(128, 14)).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
