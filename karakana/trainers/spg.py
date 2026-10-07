import os
import uuid
from collections import defaultdict, deque

import gymnasium as gym
import httpx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Categorical

from karakana.config import get_config
from karakana.metrics import evaluate_structural_geometry
from karakana.torch_loss import TorchSGOLoss
from karakana.trainers.grg import GRGController


class StructuralPolicy(nn.Module):
    """Simple MLP Policy with value head for discrete action spaces."""

    def __init__(self, obs_size, action_count):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.actor = nn.Linear(16, action_count)
        self.critic = nn.Linear(16, 1)

    def forward(self, x):
        h = self.net(x)
        return self.actor(h), self.critic(h)

    def action_probs(self, x):
        logits, _ = self.forward(x)
        return functional.softmax(logits, dim=-1)


class SPGTrainer:
    """
    Structural Policy Gradient (SPG) Trainer.
    Regularizes Policy Gradient loss with differentiable structural geometry metrics.
    Includes an Adaptive Lambda controller, Alpha-Momentum Steering, and Cloud Sync.
    """

    def __init__(
        self,
        env_name,
        lambda_sgo=None,
        lr=0.01,
        adaptive_lambda=None,
        cloud_sync=None,
        api_key="dev_key_123",
        sgo_buffer_size=None,
        entropy_coef=None,
        loss_module=None,
        tail_risk_coef=None,
        ranking_profile=None,
        grg_window_size=None,
        steering_enabled=True,
        lambda_coupling=0.003,
        coupling_k=10,
        coupling_gradient=False,
        use_coupling_trajectory=True,
        utility_power=1.1,
        num_epochs=None,
        gamma=0.99,
        **kwargs,
    ):
        if env_name not in {"HistoricalTrading-v0", "HistoricalTrading-v1"} and float(gamma) <= 0.0:
            self.gamma = 0.99
        else:
            self.gamma = float(gamma)
        self.env = gym.make(env_name)
        obs_size = self.env.observation_space.shape[0]
        action_count = self.env.action_space.n

        self.policy = StructuralPolicy(obs_size, action_count)
        self.base_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        profile_name = get_config.get_gym_env_profile_name(env_name)
        profile = get_config.get_profile(profile_name)
        env_defaults = get_config.get_gym_env_defaults(env_name)

        self.lambda_sgo = lambda_sgo if lambda_sgo is not None else float(env_defaults["lambda_sgo"])
        self.entropy_coef = entropy_coef if entropy_coef is not None else float(env_defaults["entropy_coef"])
        self.adaptive_lambda = (
            adaptive_lambda if adaptive_lambda is not None else bool(profile.get("adaptive_lambda", False))
        )
        self.cloud_sync = cloud_sync if cloud_sync is not None else bool(profile.get("cloud_sync", False))
        self.sgo_buffer_size = sgo_buffer_size if sgo_buffer_size is not None else int(env_defaults["sgo_buffer_size"])
        self.steering_enabled = steering_enabled
        loss_module = loss_module if loss_module is not None else str(env_defaults["loss_module"])
        self.loss_module = loss_module
        default_tail_risk_coef = float(env_defaults["tail_risk_coef"])
        self.tail_risk_coef = tail_risk_coef if tail_risk_coef is not None else default_tail_risk_coef

        ranking_profile = (
            ranking_profile if ranking_profile is not None else str(env_defaults.get("ranking_profile", "balanced"))
        )
        self.ranking_profile = ranking_profile

        self.sgo_loss_fn = TorchSGOLoss(
            lambda_sgo=self.lambda_sgo,
            loss_module=loss_module,
            ranking_profile=self.ranking_profile,
            lambda_coupling=lambda_coupling,
            coupling_k=coupling_k,
            coupling_gradient=coupling_gradient,
            use_coupling_trajectory=use_coupling_trajectory,
            utility_power=utility_power,
        )

        self.api_key = api_key
        self.session_id = f"spg-sync-{uuid.uuid4().hex[:8]}"

        grg_window = (
            grg_window_size if grg_window_size is not None else int(profile.get("grg", {}).get("window_size", 20))
        )

        self.grg = GRGController(self.optimizer, base_lr=lr, profile=profile_name, grg_window_size=grg_window)
        self.initial_lambda = self.lambda_sgo
        self.action_buffer_size = max(50, int(self.sgo_buffer_size) * 10)
        self.action_reward_buffer = defaultdict(lambda: deque(maxlen=self.action_buffer_size))
        self._last_alpha = 1.0
        self.last_episode_actions = []
        self.last_episode_rewards = []
        self.old_log_probs = deque(maxlen=self.action_buffer_size)
        self.num_epochs = int(num_epochs) if num_epochs is not None else None

    @property
    def current_alpha(self):
        """The structural alpha from the most recent training episode."""
        return self._last_alpha

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition has fired."""
        if not self.steering_enabled:
            return False
        return self.grg.should_stop()

    def _apply_structural_steering(self, alpha_health):
        """Dynamic adjustment of LR (via GRGController) and Lambda scaling."""
        alpha_health = (
            float(alpha_health.detach().cpu().item()) if torch.is_tensor(alpha_health) else float(alpha_health)
        )
        alpha_health = max(0.0, min(1.0, alpha_health))
        self._last_alpha = alpha_health

        # 1. LR steering delegated to GRGController
        stats = self.grg.step(alpha_health)

        # 2. Dynamic Lambda Scaling
        if alpha_health < 0.75:
            self.lambda_sgo = min(self.lambda_sgo * 1.1, 2.0)
            self.sgo_loss_fn.lambda_sgo = self.lambda_sgo
        elif stats["is_stable"] and alpha_health > 0.85:
            self.lambda_sgo = max(self.lambda_sgo * 0.95, self.initial_lambda)
            self.sgo_loss_fn.lambda_sgo = self.lambda_sgo

    @staticmethod
    def _action_tail_risk(rewards):
        """Estimate asymmetric downside severity for one action's observed rewards."""
        values = np.asarray(list(rewards), dtype=float)
        if values.size == 0:
            return 0.0
        positives = values[values > 0.0]
        negatives = values[values < 0.0]
        if positives.size == 0 or negatives.size == 0:
            return 0.0

        upside = max(float(np.mean(positives)), 1e-8)
        downside = abs(float(np.mean(negatives)))
        return max(0.0, min(1.0, downside / (downside + upside)))

    def _action_tail_risk_loss(self, log_probs, actions):
        """Penalize sampled actions that have demonstrated asymmetric downside."""
        if self.lambda_sgo <= 0 or self.tail_risk_coef <= 0:
            return None

        terms = []
        for log_prob, action in zip(log_probs, actions, strict=True):
            risk = self._action_tail_risk(self.action_reward_buffer.get(int(action), ()))
            if risk > 0.0:
                terms.append(float(risk) * log_prob)

        if not terms:
            return None
        return self.lambda_sgo * self.tail_risk_coef * torch.stack(terms).mean()

    def train_episode(self):
        obs, _ = self.env.reset()
        log_probs = []
        rewards = []
        actions = []
        obs_list = []

        done = False
        truncated = False

        while not (done or truncated):
            obs_list.append(obs)
            obs_t = torch.from_numpy(obs).float()
            logits, _ = self.policy(obs_t)
            m = Categorical(logits=logits)
            action = m.sample()

            obs, reward, done, truncated, _ = self.env.step(action.item())

            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            actions.append(int(action.item()))

        self.last_episode_actions = list(actions)
        self.last_episode_rewards = [float(r) for r in rewards]

        # 1. PPO Actor-Critic loss (replaces REINFORCE)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute PPO clip loss with actor-critic
        old_log_probs_list = [lp.detach() for lp in log_probs]

        policy_losses = []
        value_losses = []
        entropies = []
        for i, (obs_t, g, old_lp) in enumerate(zip(obs_list, returns, old_log_probs_list, strict=True)):
            obs_tensor = torch.from_numpy(obs_t).float().unsqueeze(0)
            logits, value = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor([actions[i]]))
            ratio = torch.exp(log_prob - torch.as_tensor(old_lp))
            adv = g - value.squeeze().detach()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv
            policy_losses.append(-torch.min(surr1, surr2))
            value_losses.append(functional.mse_loss(value.squeeze(), g))
            if self.entropy_coef > 0:
                entropies.append(dist.entropy().sum())

        ppo_clip_loss = torch.stack(policy_losses).mean() + 0.5 * torch.stack(value_losses).mean()
        entropy_loss = -torch.stack(entropies).mean() if entropies else torch.tensor(0.0)

        # Compute diff_outcomes for structural coupling (matching training_utils.py pattern)
        curr_alpha_health = 1.0
        diff_outcomes = torch.tensor([0.0])
        if len(log_probs) >= 5:
            with torch.no_grad():
                old_logits_list = []
                for obs_t in obs_list:
                    obs_tensor = torch.from_numpy(obs_t).float().unsqueeze(0)
                    logits, _ = self.policy(obs_tensor)
                    old_logits_list.append(logits)
                old_dists = [Categorical(logits=lg) for lg in old_logits_list]
                old_lps = [d.log_prob(torch.tensor([a])) for d, a in zip(old_dists, actions)]
                old_lps = torch.stack([lp for lp in old_lps]).squeeze()
            new_logits, _ = self.policy(torch.stack([torch.from_numpy(o).float() for o in obs_list]))
            new_dists = Categorical(logits=new_logits)
            new_lps = new_dists.log_prob(torch.tensor(actions))
            ratios = torch.exp(new_lps - old_lps.detach())
            advantages = returns - torch.tensor([0.0] * len(returns), dtype=torch.float32)
            if len(returns) > 1 and float(returns.std()) > 1e-6:
                advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            diff_outcomes = ratios * advantages

        total_loss = self.sgo_loss_fn(diff_outcomes, base_loss=ppo_clip_loss)
        if self.entropy_coef > 0:
            total_loss += self.entropy_coef * entropy_loss

        tail_risk_loss = self._action_tail_risk_loss(log_probs, actions)
        if tail_risk_loss is not None:
            total_loss += tail_risk_loss

        # Log coupling metric (rho_sq) from loss function for monitoring
        rho_attr = getattr(self.sgo_loss_fn, "last_rho_sq", None)
        self.last_rho_sq = float(rho_attr) if rho_attr is not None else 0.0

        # Compute alpha from structural geometry metrics (matching continuous_spg.py)
        curr_alpha_health = 1.0
        if len(log_probs) >= 5:
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

        # Log gradients for visualization (Task 3)
        grads = []
        for param in self.policy.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).cpu().numpy())
        if grads:
            grad_vector = np.concatenate(grads)
            output_dir = "user_data/grads"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/{self.session_id}_grads.csv", "a") as f:
                np.savetxt(f, [grad_vector], delimiter=",")

        self.optimizer.step()

        return float(sum(rewards)), float(total_loss.item())

    def _sync_lambda_with_cloud(self):
        """Poll the Cloud API for lambda suggestions based on structural trends."""
        try:
            url = "http://127.0.0.1:8000/sgo/lambda/suggest"
            stats = self.tracker.get_stats()
            payload = {
                "session_id": self.session_id,
                "current_lambda": self.lambda_sgo,
                "structural_error_history": self.tracker.history,
                "velocity": stats["velocity"],
                "momentum": stats["momentum"],
            }
            headers = {"X-API-Key": self.api_key}

            with httpx.Client(timeout=1.0) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    suggestion = response.json()
                    self.lambda_sgo = float(suggestion["recommended_lambda"])
        except Exception:
            pass
