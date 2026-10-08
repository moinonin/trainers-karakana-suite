from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from karakana_engine.torch_loss import TorchSGOLoss
from karakana_engine.trainers.grg import GRGController


class RunningMeanStd:
    """Calculates running mean and variance for normalization."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float32")
        self.var = np.ones(shape, "float32")
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ContinuousActorCritic(nn.Module):
    """Refined Actor-Critic for robotics control."""

    def __init__(self, obs_size, action_size):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu_head = nn.Linear(64, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std)
        value = self.value_head(features)
        return mu, std, value


class ContinuousSGPOTrainer:
    """
    Industrial-grade SGO-PPO Trainer with GAE and Normalization.
    Closes the gap with SB3 while maintaining structural anchors.
    """

    def __init__(
        self,
        env_name,
        lambda_sgo=0.01,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=10,
        loss_module="probability_kl",
        ranking_profile="balanced",
        steering_enabled=True,
        max_steps_per_episode=1000,
        entropy_coef=0.01,
        tail_risk_coef=0.0,
        adaptive_lambda=False,
        seed=None,
        lambda_coupling=0.003,
        coupling_k=10,
        coupling_gradient=False,
        use_coupling_trajectory=True,
        utility_power=1.1,
        num_epochs=None,
    ):
        self.env = gym.make(env_name)
        self.seed = int(seed) if seed is not None else None
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        # Action limits for clipping
        self.action_low = self.env.action_space.low.astype(np.float32)
        self.action_high = self.env.action_space.high.astype(np.float32)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.policy = ContinuousActorCritic(self.obs_size, self.action_size).to(self.device)
        self.policy_old = ContinuousActorCritic(self.obs_size, self.action_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.sgo_loss_fn = TorchSGOLoss(
            lambda_sgo=lambda_sgo,
            loss_module=loss_module,
            ranking_profile=ranking_profile,
            lambda_coupling=lambda_coupling,
            coupling_k=coupling_k,
            coupling_gradient=coupling_gradient,
            use_coupling_trajectory=use_coupling_trajectory,
            utility_power=utility_power,
        ).to(self.device)
        self.grg = GRGController(self.optimizer, base_lr=lr, profile="gym_rl", ranking_profile=ranking_profile)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.steering_enabled = steering_enabled
        self.lambda_sgo = lambda_sgo
        self.initial_lambda = lambda_sgo
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.entropy_coef = float(entropy_coef)
        self.tail_risk_coef = float(tail_risk_coef)
        self.adaptive_lambda = bool(adaptive_lambda)

        # Normalization
        self.obs_rms = RunningMeanStd(shape=(self.obs_size,))
        self.epsilon = 1e-8

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        self.buffer_values = []
        self._last_alpha = 1.0
        self.cancel_requested: Callable[[], bool] | None = None
        self.num_epochs = int(num_epochs) if num_epochs is not None else None

    @property
    def current_alpha(self):
        """The structural alpha from the most recent policy update."""
        return self._last_alpha

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition has fired."""
        if not self.steering_enabled:
            return False
        return self.grg.should_stop()

    def _apply_structural_steering(self, outcomes):
        """Dynamic adjustment of LR (via GRGController) and Lambda scaling."""
        outcomes = outcomes.detach().cpu().numpy() if torch.is_tensor(outcomes) else np.asarray(outcomes)
        pos = outcomes[outcomes > 0]
        neg = outcomes[outcomes <= 0]

        if pos.size > 0 and neg.size > 0:
            from karakana_engine.metrics import evaluate_structural_geometry

            res = evaluate_structural_geometry(pos, neg)
            alpha_health = max(0.0, min(1.0, 1.0 - float(res.get("alpha_s_sil", 1.0)) / 2.0))
            self._last_alpha = alpha_health

            # 1. LR steering delegated to GRGController
            stats = self.grg.step(alpha_health)

            # 2. Dynamic Lambda Scaling
            if self.adaptive_lambda:
                if alpha_health < 0.75:
                    self.lambda_sgo = min(self.lambda_sgo * 1.05, 1.5)
                    self.sgo_loss_fn.lambda_sgo = self.lambda_sgo
                elif stats["is_stable"] and alpha_health > 0.85:
                    self.lambda_sgo = max(self.lambda_sgo * 0.98, self.initial_lambda)
                    self.sgo_loss_fn.lambda_sgo = self.lambda_sgo

    def _normalize_obs(self, obs, *, update=True):
        if update:
            self.obs_rms.update(obs[None])
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -10, 10)

    def preprocess_observation(self, obs, *, update=False):
        normalized = self._normalize_obs(np.asarray(obs, dtype=np.float32), update=update)
        return torch.as_tensor(normalized, dtype=torch.float32, device=self.device)

    def select_action(self, state):
        state_norm = self._normalize_obs(np.asarray(state, dtype=np.float32), update=True)
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            mu, std, value = self.policy_old(state_t)
            dist = Normal(mu, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)

        self.buffer_states.append(state_t.squeeze(0))
        self.buffer_actions.append(action.squeeze(0))
        self.buffer_logprobs.append(logprob.squeeze(0))
        # value is (1,1): reduce fully to a scalar so stacks stay 1-D
        self.buffer_values.append(value.reshape(()))

        # Physics Safety: Clip continuous actions
        action_np = action.squeeze(0).cpu().numpy()
        return np.clip(action_np, self.action_low, self.action_high).astype(np.float32)

    def update(self):
        # 1. Compute GAE Returns (Differentiable flow)
        rewards = torch.tensor(self.buffer_rewards, dtype=torch.float32, device=self.device)
        masks = 1.0 - torch.tensor(self.buffer_is_terminals, dtype=torch.float32, device=self.device)

        # Use current policy values for GAE to keep graph alive if needed,
        # but standard PPO uses old values for advantage calculation.
        # To maintain differentiability for SGO, we must ensure gae_returns
        # traces back to current policy values.

        old_states = torch.stack(self.buffer_states).to(self.device).detach()
        old_actions = torch.stack(self.buffer_actions).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer_logprobs).to(self.device).detach()

        loss_val = 0
        for _ in range(self.k_epochs):
            # Forward pass to get CURRENT values for differentiable GAE
            mu, std, state_values = self.policy(old_states)
            state_values = state_values.reshape(-1)
            dist = Normal(mu, std)
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # Differentiable GAE calculation
            # We assume next_value is 0 for terminal episode-based training
            with torch.no_grad():
                old_values = torch.stack(self.buffer_values).to(self.device)

            # Use state_values (current policy) to keep gradients for SGO
            gae = 0
            returns = []
            # Bootstrap value for the state after the last stored transition.
            # Zero only when the episode truly terminated; for time-limit
            # truncation we need V(s_next) from the old policy.
            bootstrap_obs = getattr(self, "_bootstrap_obs", None)
            if bootstrap_obs is None:
                tail_vals = torch.zeros(1, device=self.device)
            else:
                with torch.no_grad():
                    nxt = torch.as_tensor(
                        self._normalize_obs(bootstrap_obs, update=False), dtype=torch.float32, device=self.device
                    )
                    _, _, nxt_v = self.policy_old(nxt.unsqueeze(0))
                self._bootstrap_obs = None
                tail_vals = nxt_v.reshape(1)
            vals = torch.cat([state_values, tail_vals])
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + self.gamma * vals[step + 1] * masks[step] - vals[step]
                gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
                returns.insert(0, gae + vals[step])

            gae_returns = torch.stack(returns)

            # PPO Logic
            ratios = torch.exp(logprobs - old_logprobs)
            # Advantages use old_values for stability but returns are differentiable
            advantages = gae_returns.detach() - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total PPO Base Loss
            returns_var = torch.var(gae_returns.detach()).clamp_min(1.0)
            value_loss = 0.5 * nn.MSELoss()(state_values, gae_returns.detach()) / returns_var
            ppo_clip_loss = -torch.min(surr1, surr2).mean() + 0.5 * value_loss
            # PPO adds `+ c_ent * H(π)` to the *objective*; minimizing the loss
            # must therefore SUBTRACT the entropy bonus or the policy collapses
            # to determinism (watching Ant's legs vibrate without walking).
            entropy_loss = -self.entropy_coef * entropy

            # Structural outcomes must be the centered advantage stream (P vs N
            # structure), not raw scaled returns which are all-positive for
            # alive-bonus envs like Ant and make the SGO geometry degenerate.
            diff_outcomes = ratios * advantages.detach()
            total_loss = self.sgo_loss_fn(diff_outcomes, base_loss=ppo_clip_loss) + entropy_loss

            if self.tail_risk_coef > 0:
                downside = torch.relu(-diff_outcomes)
                tail_risk_penalty = torch.mean(downside.square()) / (
                    torch.mean(torch.abs(diff_outcomes.detach())) + self.epsilon
                )
                total_loss = total_loss + (self.lambda_sgo * self.tail_risk_coef * tail_risk_penalty)

            # Apply Structural Steering (Task 14)
            if self.steering_enabled:
                self._apply_structural_steering(diff_outcomes - diff_outcomes.mean())

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            loss_val += total_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self._clear_buffer()
        return loss_val / self.k_epochs

    def _clear_buffer(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        self.buffer_values = []

    def train_episode(self):
        state, _ = self.env.reset()
        ep_reward = 0
        self._bootstrap_obs = None
        # Forward-displacement estimate: MuJoCo qpos dims are [z, quat(4), joints...],
        # so observation[13:] == qvel; qvel[0] is the free-joint x velocity.
        dt_env = float(getattr(getattr(self.env, "unwrapped", self.env), "dt", 0.0) or 0.0)
        qvel_start = 13
        displacement = 0.0
        for t in range(1, self.max_steps_per_episode + 1):
            if self.cancel_requested and t % 25 == 0 and self.cancel_requested():
                self._clear_buffer()
                raise RuntimeError("ContinuousSGPO training canceled during episode rollout.")
            action = self.select_action(state)
            state, reward, done, truncated, _ = self.env.step(action)
            self.buffer_rewards.append(reward)
            # Time-limit truncation is NOT a terminal state: the episode did not
            # end because the agent failed, so GAE must bootstrap with V(s_next).
            if truncated and not done:
                self.buffer_is_terminals.append(False)
                self._bootstrap_obs = np.asarray(state, dtype=np.float32)
            else:
                self.buffer_is_terminals.append(done)
            ep_reward += reward
            if dt_env > 0 and state.shape[0] > qvel_start:
                displacement += float(state[qvel_start]) * dt_env
            if done or truncated:
                break

        loss = self.update()
        return float(ep_reward), {"loss": float(loss), "displacement": round(float(displacement), 4)}
