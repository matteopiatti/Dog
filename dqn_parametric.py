# dqn_parametric.py
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple(
    "Transition", ["obs", "act_feat", "reward", "next_obs", "next_act_feat", "done"]
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


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, obs, act_feat):
        # obs: [batch, obs_dim], act_feat: [batch, act_dim]
        x = torch.cat([obs, act_feat], dim=-1)
        return self.net(x)  # [batch, 1]


class ParametricDQNAgent:
    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        act_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        buffer_capacity: int = 200_000,
        batch_size: int = 128,
        target_update_freq: int = 1000,
        device: str | None = None,
    ):
        self.player_id = player_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.qnet = QNet(obs_dim, act_dim).to(self.device)
        self.target_qnet = QNet(obs_dim, act_dim).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    def epsilon(self, eval_mode: bool) -> float:
        if eval_mode:
            return 0.0
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(
        self,
        obs: np.ndarray,
        legal_act_feats: list[np.ndarray],
        eval_mode: bool = False,
    ) -> tuple[int, np.ndarray]:
        """
        obs: 1D np array (state observation)
        legal_act_feats: list of 1D np arrays (features for each legal action)
        returns: (index_in_legal_list, chosen_act_feat)
        """
        assert len(legal_act_feats) > 0
        eps = self.epsilon(eval_mode)
        self.total_steps += 1

        if random.random() < eps:
            idx = random.randrange(len(legal_act_feats))
            return idx, legal_act_feats[idx]

        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        # build batch for all legal actions
        act_batch = torch.from_numpy(np.stack(legal_act_feats)).float().to(self.device)
        obs_batch = obs_t.repeat(len(legal_act_feats), 1)

        with torch.no_grad():
            q_values = self.qnet(obs_batch, act_batch).squeeze(1).cpu().numpy()

        idx = int(q_values.argmax())
        return idx, legal_act_feats[idx]

    def store_transition(
        self,
        obs: np.ndarray,
        act_feat: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_act_feat: np.ndarray,
        done: bool,
    ):
        self.replay.push(obs, act_feat, reward, next_obs, next_act_feat, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)

        obs_batch = torch.from_numpy(np.stack(batch.obs)).float().to(self.device)
        act_batch = torch.from_numpy(np.stack(batch.act_feat)).float().to(self.device)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_obs_batch = (
            torch.from_numpy(np.stack(batch.next_obs)).float().to(self.device)
        )
        next_act_batch = (
            torch.from_numpy(np.stack(batch.next_act_feat)).float().to(self.device)
        )
        done_batch = torch.tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Q(s,a)
        q_sa = self.qnet(obs_batch, act_batch)

        with torch.no_grad():
            q_next = self.target_qnet(next_obs_batch, next_act_batch)
            target = reward_batch + self.gamma * (1.0 - done_batch) * q_next

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "qnet": self.qnet.state_dict(),
                "target_qnet": self.target_qnet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str, map_location=None):
        device = map_location or self.device
        checkpoint = torch.load(path, map_location=device)
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(checkpoint["target_qnet"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
