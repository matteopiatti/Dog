# parametric_ppo.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class StepRecord:
    obs: np.ndarray  # [obs_dim]
    legal_act_feats: np.ndarray  # [num_legal, act_dim]
    action_idx: int  # index into legal_act_feats
    logp: float  # old log prob
    value: float  # V(s) at time of action
    reward: float  # scalar reward
    done: bool  # episode done


class ParamPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256, 256)):
        super().__init__()
        layers = []
        input_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))  # scalar logit for (s,a)
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act_feat], dim=-1)
        return self.net(x)  # [N, 1]


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(256, 256, 256)):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # [N, 1]


class ParametricPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        policy_hidden=(256, 256, 256),
        value_hidden=(256, 256, 256),
        device: str | None = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ParamPolicyNet(obs_dim, act_dim, hidden_sizes=policy_hidden).to(
            self.device
        )
        self.value_net = ValueNet(obs_dim, hidden_sizes=value_hidden).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

        self.current_episode: List[StepRecord] = []
        self.buffer: List[StepRecord] = []
        self.buffer_advantages: List[float] = []
        self.buffer_returns: List[float] = []

    # --------------------------------------------------
    # interaction
    # --------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        legal_act_feats: List[np.ndarray],
        eval_mode: bool = False,
    ) -> Tuple[int, np.ndarray, float, float]:
        """
        obs: [obs_dim] np float32
        legal_act_feats: list of [act_dim] np float32
        returns: (action_idx_in_legal_list, chosen_act_feat, logp, value)
        """
        assert len(legal_act_feats) > 0
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        value_t = self.value_net(obs_t)  # [1,1]
        value = value_t.item()

        act_arr = np.stack(legal_act_feats).astype(np.float32)
        act_t = torch.from_numpy(act_arr).float().to(self.device)  # [A, act_dim]
        obs_batch = obs_t.repeat(act_t.shape[0], 1)  # [A, obs_dim]

        with torch.no_grad():
            logits = self.policy(obs_batch, act_t).squeeze(1)  # [A]
            dist = torch.distributions.Categorical(logits=logits)
            if eval_mode:
                idx_t = torch.argmax(logits)
            else:
                idx_t = dist.sample()
            logp = dist.log_prob(idx_t).item()

        idx = int(idx_t.item())
        chosen_feat = legal_act_feats[idx]
        return idx, chosen_feat, logp, value

    def store_step(
        self,
        obs: np.ndarray,
        legal_act_feats: List[np.ndarray],
        action_idx: int,
        logp: float,
        value: float,
        reward: float,
        done: bool,
    ):
        legal_arr = np.stack(legal_act_feats).astype(np.float32)
        rec = StepRecord(
            obs=obs.astype(np.float32),
            legal_act_feats=legal_arr,
            action_idx=action_idx,
            logp=logp,
            value=value,
            reward=float(reward),
            done=bool(done),
        )
        self.current_episode.append(rec)

    def finish_episode(self):
        """Move current_episode into buffer and compute GAE/returns for it."""
        if not self.current_episode:
            return

        T = len(self.current_episode)
        rewards = [s.reward for s in self.current_episode]
        dones = [s.done for s in self.current_episode]
        values = [s.value for s in self.current_episode]

        advantages = [0.0] * T
        returns = [0.0] * T

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        self.buffer.extend(self.current_episode)
        self.buffer_advantages.extend(advantages)
        self.buffer_returns.extend(returns)
        self.current_episode = []

    # --------------------------------------------------
    # PPO update
    # --------------------------------------------------

    def _iterate_minibatches(self, indices: List[int]):
        random.shuffle(indices)
        for start in range(0, len(indices), self.minibatch_size):
            end = start + self.minibatch_size
            yield indices[start:end]

    def update(self):
        if not self.buffer:
            return

        # convert lists to numpy / tensors as needed
        N = len(self.buffer)
        advantages = np.array(self.buffer_advantages, dtype=np.float32)
        returns = np.array(self.buffer_returns, dtype=np.float32)

        # normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        indices = list(range(N))

        for _ in range(self.update_epochs):
            for mb_idx in self._iterate_minibatches(indices):
                if not mb_idx:
                    continue

                # build minibatch tensors (obs and returns/adv)
                obs_list = [self.buffer[i].obs for i in mb_idx]
                obs_t = torch.from_numpy(np.stack(obs_list)).float().to(self.device)

                adv_t = torch.from_numpy(advantages[mb_idx]).float().to(self.device)
                ret_t = torch.from_numpy(returns[mb_idx]).float().to(self.device)

                old_logp_t = torch.tensor(
                    [self.buffer[i].logp for i in mb_idx],
                    dtype=torch.float32,
                    device=self.device,
                )

                # compute new log probs and entropy sample-wise (variable-length legal sets)
                logp_new_list = []
                entropy_list = []

                for j, idx in enumerate(mb_idx):
                    step = self.buffer[idx]
                    legal_feats = step.legal_act_feats  # [A, act_dim]
                    act_idx = step.action_idx

                    legal_t = torch.from_numpy(legal_feats).float().to(self.device)
                    obs_single = (
                        obs_t[j].unsqueeze(0).repeat(legal_t.shape[0], 1)
                    )  # [A, obs_dim]

                    logits = self.policy(obs_single, legal_t).squeeze(1)  # [A]
                    dist = torch.distributions.Categorical(logits=logits)

                    a_idx_t = torch.tensor(
                        act_idx, dtype=torch.long, device=self.device
                    )
                    logp_a = dist.log_prob(a_idx_t)  # scalar
                    logp_new_list.append(logp_a)

                    entropy_list.append(dist.entropy().mean())

                logp_new_t = torch.stack(logp_new_list, dim=0)  # [B]
                entropy_t = torch.stack(entropy_list, dim=0).mean()

                # ratio
                ratio = torch.exp(logp_new_t - old_logp_t)  # [B]
                surr1 = ratio * adv_t
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                values_pred = self.value_net(obs_t).squeeze(1)  # [B]
                value_loss = 0.5 * (ret_t - values_pred).pow(2).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_t
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

        # clear buffer
        self.buffer = []
        self.buffer_advantages = []
        self.buffer_returns = []

    # --------------------------------------------------
    # checkpointing
    # --------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location=None):
        device = map_location or self.device
        checkpoint = torch.load(path, map_location=device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
