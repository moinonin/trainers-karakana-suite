import warnings
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.distributions import Categorical

try:
    import ale_py

    try:
        gym.spec("PongNoFrameskip-v4")
    except gym.error.Error:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gym.register_envs(ale_py)
except ImportError:
    ale_py = None

from karakana.torch_loss import TorchSGOLoss
from karakana.trainers.grg import GRGController
from karakana.trainers.sgpo import SGPOTrainer


class AtariTrainingError(Exception):
    """Raised when an Atari training episode receives a cancellation request."""


class FireResetEnv(gym.Wrapper):
    """
    Take action 1 (FIRE) on reset for environments like Breakout and Pong that require it to start.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.fire_action = 1 if "FIRE" in action_meanings and len(action_meanings) > 1 else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fire_action is not None:
            obs, _, done, truncated, info = self.env.step(self.fire_action)
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class AtariCNN(nn.Module):
    """
    Nature CNN architecture for Atari environments.
    """

    def __init__(self, observation_space, action_count):
        super().__init__()
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())

        self.actor = nn.Linear(512, action_count)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        elif torch.is_floating_point(x) and x.numel() and float(x.detach().max()) > 1.0:
            x = x.div(255.0)
        features = self.fc(self.cnn(x))
        return torch.softmax(self.actor(features), dim=-1), self.critic(features)


class AtariSGPOTrainer(SGPOTrainer):
    """
    Atari-optimized SGPO Trainer with CNN architecture.
    """

    def __init__(
        self,
        env_name,
        lambda_sgo=0.1,
        lr=2.5e-4,
        gamma=0.99,
        eps_clip=0.1,
        k_epochs=4,
        frame_stack=4,
        loss_module="moment",
        ranking_profile="balanced",
        lambda_coupling=0.003,
        coupling_k=10,
        coupling_gradient=False,
        use_coupling_trajectory=True,
        utility_power=1.1,
        num_epochs=None,
        steering_enabled=True,
        max_steps_per_episode=1500,
    ):
        # Initialize gym env with standard Atari preprocessing
        base_env = gym.make(env_name, render_mode="rgb_array")
        env = AtariPreprocessing(
            base_env, screen_size=84, grayscale_obs=True, frame_skip=4, terminal_on_life_loss=True
        )
        env = FireResetEnv(env)
        self.env = FrameStackObservation(env, frame_stack)

        self.action_count = self.env.action_space.n

        # Override MLP with CNN
        self.policy = AtariCNN(self.env.observation_space, self.action_count)
        self.policy_old = AtariCNN(self.env.observation_space, self.action_count)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.base_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.lambda_sgo = lambda_sgo
        self.initial_lambda = lambda_sgo
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

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.steering_enabled = steering_enabled
        self.max_steps_per_episode = max_steps_per_episode
        self.grg = GRGController(self.optimizer, base_lr=lr, profile="gym_rl", ranking_profile=ranking_profile)

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        self.cancel_requested: Callable[[], bool] | None = None
        self._last_alpha = 1.0

    @staticmethod
    def preprocess_observation(state) -> torch.Tensor:
        return torch.as_tensor(np.array(state), dtype=torch.float32).div(255.0)

    def select_action(self, state):
        with torch.no_grad():
            # State from FrameStack is [4, 84, 84] usually
            state_t = self.preprocess_observation(state).unsqueeze(0)
            probs, _ = self.policy_old(state_t)
            dist = Categorical(probs)
            action = dist.sample()
            action_scalar = action.squeeze(0)

        self.buffer_states.append(state_t.squeeze(0))
        self.buffer_actions.append(action_scalar)
        self.buffer_logprobs.append(dist.log_prob(action).squeeze(0))

        return int(action_scalar.item())

    def train_episode(self):
        """
        Runs one full episode and performs one update.
        Returns: (episode_reward, total_loss)
        """
        state, _ = self.env.reset()
        ep_reward = 0

        for t in range(1, self.max_steps_per_episode + 1):
            if self.cancel_requested and t % 25 == 0 and self.cancel_requested():
                self.buffer_states = []
                self.buffer_actions = []
                self.buffer_logprobs = []
                self.buffer_rewards = []
                self.buffer_is_terminals = []
                raise AtariTrainingError("Atari training canceled during episode rollout.")

            action = self.select_action(state)
            state, reward, done, truncated, _ = self.env.step(action)

            self.buffer_rewards.append(reward)
            self.buffer_is_terminals.append(done or truncated)

            ep_reward += reward

            if done or truncated:
                break

        # Perform update at the end of every episode for simplicity in API loop
        # and to match SPGTrainer interface
        loss_val = self.update()
        return ep_reward, loss_val

    def update(self):
        if not self.buffer_states:
            return 0.0

        # Similar to SGPOTrainer.update but ensure buffer handling for images
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer_rewards), reversed(self.buffer_is_terminals), strict=True):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        old_states = torch.stack(self.buffer_states).detach()
        old_actions = torch.stack(self.buffer_actions).detach()
        old_logprobs = torch.stack(self.buffer_logprobs).detach()

        # Release rollout buffer immediately to keep memory usage low
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []

        total_loss_val = 0.0
        for _ in range(self.k_epochs):
            probs, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)
            dist = Categorical(probs)
            logprobs = dist.log_prob(old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            ppo_clip_loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values, rewards)
            diff_outcomes = ratios * advantages.detach()

            total_loss = self.sgo_loss_fn(diff_outcomes, base_loss=ppo_clip_loss)

            # Updates _last_alpha; grg and lambda scaling are guarded by steering_enabled internally
            self._apply_structural_steering(diff_outcomes)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss_val = total_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        del old_states, old_actions, old_logprobs, rewards
        return total_loss_val

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition has fired and steering is enabled."""
        if not self.steering_enabled:
            return False
        return self.grg.should_stop()


if __name__ == "__main__":
    # Test initialization
    print("Testing AtariSGPOTrainer initialization...")
    try:
        trainer = AtariSGPOTrainer("PongNoFrameskip-v4")
        print("Success: AtariSGPOTrainer initialized with CNN.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Note: Requires gymnasium[atari] and shimmy[atari].")
