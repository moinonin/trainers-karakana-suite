import ast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from karakana_engine.config import get_config
from karakana_engine.metrics import evaluate_structural_geometry
from karakana_engine.torch_loss import TorchSGOLoss
from karakana_engine.trainers.grg import GRGController


class ActorCritic(nn.Module):
    def __init__(self, obs_size, action_count):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_count),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(nn.Linear(obs_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.actor(x), self.critic(x)


class SGPOTrainer:
    """
    Structural Geometry Policy Optimization (SGPO) Trainer.

    Regularizes PPO clip objective with differentiable structural geometry.
    Includes real-time Alpha-Momentum steering.

    Supports two modes:
      1. Gym env mode (default): env_name is a Gym env spec string.
      2. Dataset mode: pass states/actions/rewards via the `data` parameter
         (list of dicts with keys 'state', 'action', 'reward', 'done').
         In this mode, env_name should be None and observation/action space
         dimensions must be specified explicitly.

    Examples:
        # Gym env mode
        trainer = SGPOTrainer("CartPole-v1", lambda_sgo=0.2)

        # Dataset mode (generic — works with any dataset)
        trainer = SGPOTrainer(
            env_name=None,
            obs_size=4,
            action_count=2,
            data=my_dataset,  # list of dicts or pandas DataFrame
            lambda_sgo=0.2,
        )
        trainer.train_dataset(num_episodes=100)
    """

    def __init__(
        self,
        env_name=None,
        lambda_sgo=None,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.20,
        k_epochs=4,
        loss_module=None,
        steering_enabled=True,
        lambda_coupling=0.003,
        use_coupling_trajectory=True,
        coupling_k=10,
        coupling_gradient=True,
        # Dataset mode parameters (alternative to env_name)
        obs_size=None,
        action_count=None,
        data=None,
    ):
        # ── Mode selection: Gym env vs. dataset ──
        if env_name is not None:
            # Gym env mode
            self.env = gym.make(env_name)
            if isinstance(self.env.action_space, gym.spaces.Box):
                raise TypeError(
                    f"SGPOTrainer requires Discrete action space. For continuous environments like '{env_name}', use ContinuousSGPOTrainer."
                )
            obs_size = self.env.observation_space.shape[0]
            action_count = self.env.action_space.n
            self.dataset_mode = False
        else:
            # Dataset mode — no Gym env
            self.env = None
            self.dataset_mode = True
            if obs_size is None or action_count is None:
                raise ValueError(
                    "Dataset mode requires obs_size and action_count to be specified explicitly when env_name is None."
                )

        self.obs_size = obs_size
        self.action_count = action_count
        self.data = data  # May be None (Gym mode) or a list/dict/DataFrame (dataset mode)

        self.policy = ActorCritic(obs_size, action_count)
        self.base_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_size, action_count)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.lambda_sgo = lambda_sgo if lambda_sgo is not None else get_config.get("lambda_sgo", 0.1)
        self.initial_lambda = self.lambda_sgo
        l_mod = loss_module if loss_module is not None else get_config.get("loss_module", "moment")
        self.sgo_loss_fn = TorchSGOLoss(
            lambda_sgo=self.lambda_sgo,
            loss_module=l_mod,
            lambda_coupling=lambda_coupling,
            use_coupling_trajectory=use_coupling_trajectory,
            coupling_k=coupling_k,
            coupling_gradient=coupling_gradient,
        )

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.steering_enabled = steering_enabled
        self.grg = GRGController(self.optimizer, base_lr=lr, profile="gym_rl" if not self.dataset_mode else "dataset")

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        self._last_alpha = 1.0

    @property
    def current_alpha(self):
        """The structural alpha from the most recent policy update."""
        return self._last_alpha

    def _apply_structural_steering(self, outcomes):
        """Dynamic adjustment of LR (via GRGController) and Lambda scaling."""
        outcomes = outcomes.detach().cpu().numpy() if torch.is_tensor(outcomes) else np.asarray(outcomes)
        pos = outcomes[outcomes > 0]
        neg = outcomes[outcomes <= 0]

        if pos.size > 0 and neg.size > 0:
            res = evaluate_structural_geometry(pos, neg)
            alpha_health = max(0.0, min(1.0, 1.0 - float(res.get("alpha_s_sil", 1.0)) / 2.0))
            self._last_alpha = alpha_health
            if not self.steering_enabled:
                return

            # 1. LR steering delegated to GRGController
            stats = self.grg.step(alpha_health)

            # 2. Dynamic Lambda Scaling
            if alpha_health < 0.75:
                self.lambda_sgo = min(self.lambda_sgo * 1.05, 1.5)
                self.sgo_loss_fn.lambda_sgo = self.lambda_sgo
            elif stats["is_stable"] and alpha_health > 0.85:
                self.lambda_sgo = max(self.lambda_sgo * 0.98, self.initial_lambda)
                self.sgo_loss_fn.lambda_sgo = self.lambda_sgo

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition has fired and steering is enabled."""
        if not self.steering_enabled:
            return False
        return self.grg.should_stop()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            probs, _ = self.policy_old(state)
            dist = Categorical(probs)
            action = dist.sample()

        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_logprobs.append(dist.log_prob(action))

        return action.item()

    def update(self):
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer_rewards), reversed(self.buffer_is_terminals), strict=True):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if len(rewards) > 1 else rewards - rewards.mean()

        # convert list to tensor
        old_states = torch.stack(self.buffer_states).detach()
        old_actions = torch.stack(self.buffer_actions).detach()
        old_logprobs = torch.stack(self.buffer_logprobs).detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values = self.evaluate(old_states, old_actions)

            # 1. Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 2. Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

            # Final PPO loss
            base_ppo_loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards)

            # Differentiable outcomes for SGO regularization.
            # The whitepaper (eq. 8.4) defines z_t = q_t(θ) * Ĝ_t where q_t is
            # the likelihood ratio and Ĝ_t is a detached GAE return. The gradient
            # must flow through the actor's probability output to be useful.
            #
            # We use ratios * advantages.detach() as the outcome signal:
            #   - Positive advantage → positive outcome (policy was right to take this action)
            #   - Negative advantage → negative outcome (policy was wrong)
            #   - Gradient flows through ratios → actor parameters
            # This replaces the old StructuralTrap-specific branch and the
            # state_values fallback, both of which failed to carry an actor gradient.
            diff_outcomes = ratios * advantages.detach()

            # Apply Structural Steering (Task 14)
            if self.steering_enabled:
                self._apply_structural_steering(diff_outcomes - diff_outcomes.mean())

            # take gradient step
            total_loss = (base_ppo_loss + self.sgo_loss_fn(diff_outcomes - diff_outcomes.mean(), base_ppo_loss)).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []

    def train_dataset(self, num_episodes=100, update_every=1, max_steps_per_episode=None):
        """
        Train on a pre-collected dataset (dataset mode).

        Iterates through the dataset `num_episodes` times, collecting
        transitions in the PPO buffer and calling update() every
        `update_every` episodes.

        Args:
            num_episodes: Number of passes through the dataset.
            update_every: Call update() every N episodes.
            max_steps_per_episode: Max steps per episode. If None, uses full dataset.

        Returns:
            List of total rewards per episode.
        """
        if not self.dataset_mode:
            raise RuntimeError("train_dataset() is only available in dataset mode (env_name=None).")
        if self.data is None:
            raise RuntimeError("No data provided. Pass `data` to SGPOTrainer() constructor.")

        rewards_history = []

        for episode in range(num_episodes):
            # Reset buffers for this episode
            self.buffer_states = []
            self.buffer_actions = []
            self.buffer_logprobs = []
            self.buffer_rewards = []
            self.buffer_is_terminals = []

            total_reward = 0.0
            steps = 0

            if isinstance(self.data, dict):
                # Dict with keys 'states', 'actions', 'rewards'
                states = self.data["states"]
                actions = self.data["actions"]
                rewards = self.data["rewards"]
                n = len(rewards)
                limit = max_steps_per_episode or n
                for i in range(min(limit, n)):
                    s = states[i]
                    a = actions[i]
                    r = rewards[i]

                    state_t = torch.from_numpy(np.asarray(s)).float()
                    action_t = torch.tensor([a], dtype=torch.long)

                    with torch.no_grad():
                        probs, _ = self.policy_old(state_t.unsqueeze(0))
                        dist = Categorical(probs)
                        logprob = dist.log_prob(action_t)

                    self.buffer_states.append(state_t)
                    self.buffer_actions.append(action_t)
                    self.buffer_logprobs.append(logprob)
                    self.buffer_rewards.append(float(r))
                    self.buffer_is_terminals.append(False)

                    total_reward += float(r)
                    steps += 1

                # Mark last step as terminal
                if self.buffer_is_terminals:
                    self.buffer_is_terminals[-1] = True

            elif isinstance(self.data, (list, tuple)):
                # List of dicts: [{'state': ..., 'action': ..., 'reward': ..., 'done': ...}]
                limit = max_steps_per_episode or len(self.data)
                for i in range(min(limit, len(self.data))):
                    row = self.data[i]
                    s = row["state"]
                    a = row["action"]
                    r = row["reward"]
                    d = row.get("done", False)

                    state_t = torch.from_numpy(np.asarray(s)).float()
                    action_t = torch.tensor([a], dtype=torch.long)

                    with torch.no_grad():
                        probs, _ = self.policy_old(state_t.unsqueeze(0))
                        dist = Categorical(probs)
                        logprob = dist.log_prob(action_t)

                    self.buffer_states.append(state_t)
                    self.buffer_actions.append(action_t)
                    self.buffer_logprobs.append(logprob)
                    self.buffer_rewards.append(float(r))
                    self.buffer_is_terminals.append(bool(d))

                    total_reward += float(r)
                    steps += 1

            elif hasattr(self.data, "iloc"):  # pandas DataFrame
                limit = max_steps_per_episode or len(self.data)
                for i in range(min(limit, len(self.data))):
                    row = self.data.iloc[i]
                    s = row["state"]
                    a = row["action"]
                    r = row["reward"]
                    d = row.get("done", False)

                    if isinstance(s, str):
                        s = np.array(ast.literal_eval(s), dtype=np.float32)
                    else:
                        s = np.asarray(s, dtype=np.float32)

                    state_t = torch.from_numpy(s).float()
                    action_t = torch.tensor([int(a)], dtype=torch.long)

                    with torch.no_grad():
                        probs, _ = self.policy_old(state_t.unsqueeze(0))
                        dist = Categorical(probs)
                        logprob = dist.log_prob(action_t)

                    self.buffer_states.append(state_t)
                    self.buffer_actions.append(action_t)
                    self.buffer_logprobs.append(logprob)
                    self.buffer_rewards.append(float(r))
                    self.buffer_is_terminals.append(bool(d))

                    total_reward += float(r)
                    steps += 1

            else:
                raise ValueError(f"Unsupported data type: {type(self.data)}")

            # Update policy
            self.update()
            rewards_history.append(float(total_reward))

            if (episode + 1) % 10 == 0:
                print(f"Dataset episode {episode + 1:3d} | Reward: {total_reward:8.2f} | Steps: {steps}")

        return rewards_history

    def reset(self):
        """
        Reset the environment (Gym mode) or dataset iteration state (dataset mode).
        In dataset mode, this is a no-op since data is pre-collected.
        """
        if self.env is not None:
            return self.env.reset()

    def evaluate(self, state, action):
        probs, state_values = self.policy(state)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(action)
        return action_logprobs, torch.squeeze(state_values)

    def train_episode(self):
        """Run one episode: collect buffer (via select_action), call update(), return total reward."""
        total_reward = 0.0
        state, _ = self.env.reset()
        done = False
        truncated = False
        # Clear buffers first; select_action fills them.
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        while not (done or truncated):
            action = self.select_action(state)
            # Note: select_action already appended state/action/logprob to buffers.
            # We only need to step the env and append reward/terminal here.
            next_state, reward, done, truncated, info = self.env.step(action)
            self.buffer_rewards.append(reward)
            self.buffer_is_terminals.append(done or truncated)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        self.update()
        return float(total_reward)


def run_sgpo_demo():
    print("--- SGPO: Structural PPO Demo (CartPole) ---")
    trainer = SGPOTrainer("CartPole-v1", lambda_sgo=0.2)

    max_episodes = 50
    update_timestep = 200  # Update policy every 200 steps
    time_step = 0

    for i_episode in range(1, max_episodes + 1):
        state, _ = trainer.env.reset()
        current_ep_reward = 0
        for _t in range(1, 500):
            time_step += 1
            action = trainer.select_action(state)
            state, reward, done, truncated, _ = trainer.env.step(action)

            trainer.buffer_rewards.append(reward)
            trainer.buffer_is_terminals.append(done or truncated)

            current_ep_reward += reward

            if time_step % update_timestep == 0:
                trainer.update()

            if done or truncated:
                break

        if i_episode % 10 == 0:
            print(f"Episode {i_episode:3d} | Reward: {current_ep_reward:5.1f}")

    print("\nSUCCESS: SGPO Training cycle complete.")


if __name__ == "__main__":
    run_sgpo_demo()
