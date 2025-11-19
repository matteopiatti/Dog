# dqn_agent.py
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        num_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        device: str | None = None,
    ):
        self.player_id = player_id
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQNNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = DQNNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.replay = ReplayBuffer(buffer_capacity)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    def epsilon(self):
        # linear decay
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(
        self, obs: np.ndarray, legal_actions: list[int], eval_mode: bool = False
    ) -> int:
        # obs: 1D np array
        if len(legal_actions) == 0:
            return 0  # dummy, env should not ask for action here

        eps = 0.0 if eval_mode else self.epsilon()
        self.total_steps += 1

        if random.random() < eps:
            return random.choice(legal_actions)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(obs_t)[0].cpu().numpy()

        # mask illegal actions by -inf
        masked_q = np.full_like(q_values, -1e9)
        masked_q[legal_actions] = q_values[legal_actions]
        return int(masked_q.argmax())

    def store_transition(self, state, action, next_state, reward, done):
        self.replay.push(state, action, next_state, reward, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)

        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        action_batch = torch.tensor(
            batch.action, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_state_batch = (
            torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        )
        done_batch = torch.tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Q(s,a)
        q_values = self.online_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # max_a' Q_target(s', a')
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target = reward_batch + self.gamma * (1.0 - done_batch) * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # target network update
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
