from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Normal

from karakana.metrics import AlphaMomentumTracker, evaluate_structural_geometry
from karakana.torch_loss import TorchSGOLoss


class StructuralGaussianPolicy(nn.Module):
    """Gaussian Policy with value head for continuous action spaces."""

    def __init__(self, obs_size, action_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        x = self.net(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        value = self.value_head(x).squeeze(-1)
        return mu, std, value


class ContinuousSPGTrainer:
    """
    Structural Policy Gradient Trainer for Continuous Action Spaces.
    Designed for MuJoCo and other continuous control tasks.
    """

    def __init__(
        self,
        env_name,
        lambda_sgo=0.1,
        lr=3e-4,
        adaptive_lambda=False,
        steering_enabled=True,
        entropy_coef=0.01,
        loss_module="moment",
        ranking_profile="balanced",
        lambda_coupling=0.003,
        coupling_k=10,
        coupling_gradient=False,
        use_coupling_trajectory=True,
        utility_power=1.1,
        num_epochs=None,
        gamma=0.99,
        **kwargs,
    ):
        self.gamma = float(gamma)
        self.env = gym.make(env_name)
        obs_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]

        self.policy = StructuralGaussianPolicy(obs_size, action_size)
        self.base_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.sgo_loss_fn = TorchSGOLoss(
            lambda_sgo=lambda_sgo,
            loss_module=loss_module,
            ranking_profile=ranking_profile,
            lambda_coupling=lambda_coupling,
            coupling_k=coupling_k,
            coupling_gradient=coupling_gradient,
            use_coupling_trajectory=use_coupling_trajectory,
            utility_power=utility_power,
        )
        self.lambda_sgo = lambda_sgo
        self.loss_module = loss_module
        self.entropy_coef = entropy_coef

        self.adaptive_lambda = adaptive_lambda
        self.steering_enabled = steering_enabled
        self.alpha_tracker = AlphaMomentumTracker(window_size=15)
        self.initial_lambda = lambda_sgo
        self._last_alpha = 1.0
        self.old_log_probs = deque(maxlen=1000)
        self.num_epochs = int(num_epochs) if num_epochs is not None else None

    @property
    def current_alpha(self):
        """The structural alpha from the most recent training episode."""
        return self._last_alpha

    def should_stop(self) -> bool:
        """Return True if the AlphaMomentumTracker e-stop condition has fired."""
        if not self.steering_enabled:
            return False
        return self.alpha_tracker.should_stop()

    def _apply_structural_steering(self, alpha_health):
        """Dynamic adjustment of LR and Lambda based on Alpha-Drift Velocity."""
        alpha_health = max(0.0, min(1.0, float(alpha_health)))
        self._last_alpha = alpha_health
        self.alpha_tracker.push(alpha_health)
        stats = self.alpha_tracker.get_stats()
        v_alpha = stats["v_alpha"]

        # 1. Structural LR Scheduling
        if v_alpha < -0.01:
            new_lr = self.base_lr * 0.1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
        elif v_alpha > 0.005:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.base_lr

        # 2. Dynamic Lambda Scaling
        if alpha_health < 0.75:
            self.sgo_loss_fn.lambda_sgo = min(self.sgo_loss_fn.lambda_sgo * 1.05, 1.5)
        elif stats["is_stable"] and alpha_health > 0.85:
            self.sgo_loss_fn.lambda_sgo = max(self.sgo_loss_fn.lambda_sgo * 0.98, self.initial_lambda)

    def train_episode(self):
        obs, _ = self.env.reset()
        log_probs = []
        rewards = []
        actions = []
        obs_list = []

        done = False
        truncated = False

        while not (done or truncated):
            obs_t = torch.from_numpy(obs).float()
            mu, std, value = self.policy(obs_t)
            dist = Normal(mu, std)
            action = dist.sample()

            obs, reward, done, truncated, _ = self.env.step(action.numpy())

            log_probs.append(dist.log_prob(action).sum())
            rewards.append(reward)
            actions.append(action.clone())
            obs_list.append(obs.copy())

        self.last_obs = obs

        # 1. PPO Actor-Critic loss (replaces REINFORCE)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Frozen old log probs (populate from current episode if first call)
        old_lps = [lp.detach() for lp in log_probs]

        policy_losses = []
        value_losses = []
        entropies = []
        for i, (obs_t, g, old_lp) in enumerate(zip(obs_list, returns, old_lps, strict=True)):
            obs_tensor = torch.from_numpy(obs_t).float()
            mu, std, value = self.policy(obs_tensor)
            dist = Normal(mu, std)
            log_prob = dist.log_prob(actions[i]).sum()
            ratio = torch.exp(log_prob - old_lp)
            surr1 = ratio * g
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * g
            policy_losses.append(-torch.min(surr1, surr2))
            value_losses.append(functional.mse_loss(value, g))
            entropies.append(dist.entropy().sum())

        ppo_clip_loss = torch.stack(policy_losses).mean() + 0.5 * torch.stack(value_losses).mean()
        entropy_loss = -torch.stack(entropies).mean() if entropies else torch.tensor(0.0, dtype=torch.float32)

        # 2. Structural Policy Gradient with PPO coupling (matching training_utils.py)
        ep_reward = sum(rewards)
        curr_alpha_health = 1.0
        diff_outcomes = torch.tensor([0.0], dtype=torch.float32)
        total_loss = ppo_clip_loss + self.entropy_coef * entropy_loss
        if self.lambda_sgo > 0 and len(log_probs) > 1:
            with torch.no_grad():
                obs_tensors = [torch.from_numpy(o).float() for o in obs_list]
                old_lps_tensor = torch.stack([torch.as_tensor(lp, dtype=torch.float32) for lp in old_lps])
            new_lps = []
            for obs_t, action in zip(obs_tensors, actions, strict=True):
                mu, std, _ = self.policy(obs_t)
                dist = Normal(mu, std)
                new_lps.append(dist.log_prob(action).sum())
            new_lps = torch.stack(new_lps)
            ratios = torch.exp(new_lps - old_lps_tensor.detach())
            advantages = returns
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            diff_outcomes = ratios * advantages

            total_loss = self.sgo_loss_fn(diff_outcomes, base_loss=ppo_clip_loss) + self.entropy_coef * entropy_loss

            detached_outcomes = diff_outcomes.detach().cpu().numpy()
            positives = detached_outcomes[detached_outcomes > 0]
            negatives = detached_outcomes[detached_outcomes <= 0]
            if positives.size > 0 and negatives.size > 0:
                res = evaluate_structural_geometry(positives, negatives)
                curr_alpha_health = max(0.0, min(1.0, 1.0 - float(res.get("alpha_s_sil", 1.0)) / 2.0))
            else:
                curr_alpha_health = 0.5
        elif self.lambda_sgo > 0:
            curr_alpha_health = 0.5

        # Apply Structural Steering (Task 14)
        self._last_alpha = curr_alpha_health
        if self.steering_enabled:
            self._apply_structural_steering(curr_alpha_health)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return float(ep_reward), {"loss": float(total_loss.item()), "terminated": True}


if __name__ == "__main__":
    # Test on Pendulum-v1 (Continuous classic control)
    print("--- Continuous SPG Test: Pendulum-v1 ---")
    trainer = ContinuousSPGTrainer("Pendulum-v1", lambda_sgo=0.2)
    for i in range(10):
        r, loss_val = trainer.train_episode()
        print(f"Episode {i + 1} | Reward: {r:7.1f}")
