from collections import defaultdict, deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence

from karakana_engine.config import get_config
from karakana_engine.metrics import evaluate_structural_geometry
from karakana_engine.torch_loss import TorchSGOLoss
from karakana_engine.trainers.grg import GRGController


class GRPORefPolicy(nn.Module):
    """
    Actor policy for Group Relative Policy Optimization (GRPO).
    Notice: No critic / value head is needed!
    """

    def __init__(self, obs_size: int, action_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return functional.softmax(logits, dim=-1)


class ContinuousGRPORefPolicy(nn.Module):
    """
    Gaussian Actor policy for continuous Group Relative Policy Optimization (GRPO).
    Outputs mean (mu) and standard deviation (std). No critic / value head is needed!
    """

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(x)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std)
        return mu, std


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer with SGO Regularization.

    Eliminates the learned critic (V(s)) network entirely.
    Samples a group of G rollouts per training step, computes the normalized
    advantage relative to the group mean and standard deviation:
        A_i = (R_i - mean(R)) / (std(R) + eps)
    Optimizes clipped surrogate objective with an explicit KL penalty to reference weights,
    coupled with differentiable Structural Geometry (TorchSGOLoss) and GRG steering.
    """

    def __init__(
        self,
        env_name: str,
        lambda_sgo: float | None = None,
        lr: float = 0.01,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.04,
        entropy_coef: float = 0.01,
        tail_risk_coef: float = 0.0,
        loss_module: str | None = None,
        ranking_profile: str | None = None,
        steering_enabled: bool = True,
        lambda_coupling: float = 0.003,
        coupling_k: int = 10,
        coupling_gradient: bool = False,
        use_coupling_trajectory: bool = True,
        utility_power: float = 1.1,
        gamma: float = 0.99,
        **kwargs: Any,
    ):
        self.env_name = env_name
        self.env = gym.make(env_name)
        obs_size = self.env.observation_space.shape[0]
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)

        if self.is_continuous:
            action_dim = self.env.action_space.shape[0]
            self.action_size = action_dim
            self.action_count = action_dim
            self.action_low = self.env.action_space.low.astype(np.float32)
            self.action_high = self.env.action_space.high.astype(np.float32)
            self.policy = ContinuousGRPORefPolicy(obs_size, action_dim)
            self.ref_policy = ContinuousGRPORefPolicy(obs_size, action_dim)
        elif hasattr(self.env.action_space, "n"):
            action_count = self.env.action_space.n
            self.action_count = action_count
            self.action_size = action_count
            self.policy = GRPORefPolicy(obs_size, action_count)
            self.ref_policy = GRPORefPolicy(obs_size, action_count)
        else:
            raise TypeError(f"Unsupported action space: {type(self.env.action_space)}")

        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()

        self.base_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.group_size = max(2, int(group_size))
        self.clip_eps = float(clip_eps)
        self.kl_coef = float(kl_coef)
        self.entropy_coef = float(entropy_coef)
        self.tail_risk_coef = float(tail_risk_coef)
        self.steering_enabled = steering_enabled
        self.gamma = float(gamma)

        self.lambda_sgo = (
            lambda_sgo
            if lambda_sgo is not None
            else get_config.get_gym_env_defaults(env_name).get("lambda_sgo", 0.05)
        )
        self.loss_module = (
            loss_module
            if loss_module is not None
            else get_config.get_gym_env_defaults(env_name).get("loss_module", "probability_kl")
        )

        self.sgo_loss_fn = TorchSGOLoss(
            lambda_sgo=self.lambda_sgo,
            loss_module=self.loss_module,
            ranking_profile=ranking_profile or "balanced",
            lambda_coupling=lambda_coupling,
            coupling_k=coupling_k,
            coupling_gradient=coupling_gradient,
            use_coupling_trajectory=use_coupling_trajectory,
            utility_power=utility_power,
        )

        self.initial_lambda = self.lambda_sgo
        self.grg = GRGController(self.optimizer, base_lr=lr, profile="gym_rl")
        self._last_alpha = 1.0
        self.last_rho_sq = 0.0
        self.last_episode_rewards: list[float] = []
        self.last_episode_actions: list[Any] = []
        self.action_reward_buffer: dict[Any, deque[float]] = defaultdict(lambda: deque(maxlen=200))

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition has fired."""
        if not self.steering_enabled:
            return False
        return self.grg.should_stop()

    @property
    def current_alpha(self) -> float:
        return self._last_alpha

    def _rollout_single(self) -> dict[str, Any]:
        """Rollout a single trajectory under current policy."""
        obs, _ = self.env.reset()
        obs_list = []
        actions = []
        rewards = []
        log_probs = []

        done = False
        truncated = False
        while not (done or truncated):
            obs_list.append(obs)
            obs_t = torch.from_numpy(obs).float()
            if self.is_continuous:
                with torch.no_grad():
                    mu, std = self.policy(obs_t)
                    dist = Normal(mu, std)
                    action = dist.sample()
                    lp = dist.log_prob(action).sum()

                action_np = action.cpu().numpy()
                action_clipped = np.clip(action_np, self.action_low, self.action_high)
                next_obs, reward, done, truncated, _ = self.env.step(action_clipped)
                actions.append(action_clipped)
            else:
                with torch.no_grad():
                    logits = self.policy(obs_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    lp = dist.log_prob(action)

                next_obs, reward, done, truncated, _ = self.env.step(action.item())
                actions.append(int(action.item()))

            rewards.append(float(reward))
            log_probs.append(lp)
            obs = next_obs

        # Compute discounted return
        discounted_return = 0.0
        for r in reversed(rewards):
            discounted_return = r + self.gamma * discounted_return

        total_reward = sum(rewards)
        if not self.is_continuous:
            for a, r in zip(actions, rewards, strict=True):
                self.action_reward_buffer[a].append(r)

        return {
            "obs_list": obs_list,
            "actions": actions,
            "rewards": rewards,
            "total_reward": total_reward,
            "discounted_return": discounted_return,
            "old_log_probs": torch.stack(log_probs),
        }

    def train_episode(self) -> tuple[float, float]:
        """
        Execute one GRPO training iteration over a group of rollouts.
        Returns: (mean_reward across group, loss value).
        """
        # 1. Collect group of rollouts
        trajectories = [self._rollout_single() for _ in range(self.group_size)]
        group_returns = np.array([t["discounted_return"] for t in trajectories], dtype=np.float32)
        group_rewards = [t["total_reward"] for t in trajectories]

        self.last_episode_rewards = list(group_rewards)
        self.last_episode_actions = list(trajectories[0]["actions"])
        mean_reward = float(np.mean(group_rewards))

        # 2. Compute Group Relative Advantages
        ret_mean = float(np.mean(group_returns))
        ret_std = float(np.std(group_returns))
        if ret_std < 1e-6:
            group_advantages = np.zeros_like(group_returns)
        else:
            group_advantages = (group_returns - ret_mean) / (ret_std + 1e-8)

        # 3. Policy update across trajectories in the group
        surr_losses = []
        kl_losses = []
        entropies = []
        diff_outcome_list = []

        for i, traj in enumerate(trajectories):
            adv_scalar = float(group_advantages[i])
            if not traj["obs_list"]:
                continue

            obs_batch = torch.from_numpy(np.array(traj["obs_list"], dtype=np.float32)).float()
            old_lp = traj["old_log_probs"]

            if self.is_continuous:
                actions_batch = torch.from_numpy(np.array(traj["actions"], dtype=np.float32)).float()
                mu, std = self.policy(obs_batch)
                dist = Normal(mu, std)
                new_lp = dist.log_prob(actions_batch).sum(dim=-1)
            else:
                actions_batch = torch.tensor(traj["actions"], dtype=torch.long)
                logits = self.policy(obs_batch)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(actions_batch)

            # Ratio & Clipped objective
            ratios = torch.exp(new_lp - old_lp.detach())
            adv_tensor = torch.full_like(ratios, adv_scalar)
            surr1 = ratios * adv_tensor
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_tensor
            surr_losses.append(-torch.min(surr1, surr2).mean())

            # Reference KL divergence: KL(pi_theta || pi_ref)
            if self.is_continuous:
                with torch.no_grad():
                    ref_mu, ref_std = self.ref_policy(obs_batch)
                    ref_dist = Normal(ref_mu, ref_std)
                kl = kl_divergence(dist, ref_dist).sum(dim=-1).mean()
                kl_losses.append(kl)
                if self.entropy_coef > 0:
                    entropies.append(dist.entropy().sum(dim=-1).mean())
            else:
                with torch.no_grad():
                    ref_logits = self.ref_policy(obs_batch)
                    ref_dist = Categorical(logits=ref_logits)
                # Analytical KL for categorical
                p = dist.probs
                log_p = dist.logits - dist.logits.logsumexp(dim=-1, keepdim=True)
                log_ref_p = ref_dist.logits - ref_dist.logits.logsumexp(dim=-1, keepdim=True)
                kl = (p * (log_p - log_ref_p)).sum(dim=-1).mean()
                kl_losses.append(kl)
                if self.entropy_coef > 0:
                    entropies.append(dist.entropy().mean())

            # Differentiable outcomes for SGO coupling
            diff_outcome_list.append(ratios * adv_tensor)

        base_grpo_loss = torch.stack(surr_losses).mean() if surr_losses else torch.tensor(0.0)
        total_kl = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0)
        total_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)

        # 4. SGO Regularization & Coupling
        if diff_outcome_list:
            diff_outcomes = torch.cat(diff_outcome_list)
        else:
            diff_outcomes = torch.tensor([0.0])

        total_loss = self.sgo_loss_fn(diff_outcomes, base_loss=base_grpo_loss)
        total_loss += self.kl_coef * total_kl
        if self.entropy_coef > 0:
            total_loss -= self.entropy_coef * total_entropy

        # Log coupling metric
        rho_attr = getattr(self.sgo_loss_fn, "last_rho_sq", None)
        self.last_rho_sq = float(rho_attr) if rho_attr is not None else 0.0

        # 5. Structural Alpha Evaluation & Steering
        curr_alpha_health = 1.0
        if len(diff_outcomes) >= 5:
            detached = diff_outcomes.detach().cpu().numpy()
            pos = detached[detached > 0]
            neg = detached[detached <= 0]
            if pos.size > 0 and neg.size > 0:
                res = evaluate_structural_geometry(pos, neg)
                curr_alpha_health = max(0.0, min(1.0, 1.0 - float(res.get("alpha_s_sil", 1.0)) / 2.0))
            else:
                curr_alpha_health = 0.5
        elif self.lambda_sgo > 0:
            curr_alpha_health = 0.5

        self._last_alpha = curr_alpha_health
        if self.steering_enabled:
            # Alpha-momentum velocity steering via GRGController
            stats = self.grg.step(curr_alpha_health)
            if curr_alpha_health < 0.75:
                self.lambda_sgo = min(self.lambda_sgo * 1.1, 2.0)
                self.sgo_loss_fn.lambda_sgo = self.lambda_sgo
            elif stats.get("is_stable") and curr_alpha_health > 0.85:
                self.lambda_sgo = max(self.lambda_sgo * 0.95, self.initial_lambda)
                self.sgo_loss_fn.lambda_sgo = self.lambda_sgo

        # 6. Backward and step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Slowly track reference policy (Polyak average or periodic update)
        with torch.no_grad():
            for p_ref, p_curr in zip(self.ref_policy.parameters(), self.policy.parameters(), strict=True):
                p_ref.data.copy_(0.95 * p_ref.data + 0.05 * p_curr.data)

        return mean_reward, float(total_loss.item())
