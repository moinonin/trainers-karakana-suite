"""Tests for basic Gymnasium environment integration with SGO metrics."""

import gymnasium as gym
import numpy as np
import pytest

from karakana.metrics import evaluate_structural_geometry

ENV_NAMES = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]


class TestGymEnvironments:
    @pytest.mark.parametrize("env_name", ENV_NAMES)
    def test_env_initialises(self, env_name):
        env = gym.make(env_name)
        assert env is not None
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_name", ENV_NAMES)
    def test_env_step_returns_valid_types(self, env_name):
        env = gym.make(env_name)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_name", ENV_NAMES)
    def test_env_observation_space_shape(self, env_name):
        env = gym.make(env_name)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        env.close()

    @pytest.mark.parametrize("env_name", ENV_NAMES)
    def test_action_space_samples_are_valid(self, env_name):
        env = gym.make(env_name)
        for _ in range(10):
            action = env.action_space.sample()
            assert action in range(env.action_space.n)
        env.close()

    @pytest.mark.parametrize("env_name", ENV_NAMES)
    def test_short_rollout(self, env_name):
        env = gym.make(env_name)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        assert steps > 0
        env.close()

    @pytest.mark.parametrize("env_name", ENV_NAMES)
    def test_evaluate_structural_geometry_on_rollout(self, env_name):
        env = gym.make(env_name)
        successes = []
        failures = []
        for _ in range(5):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated

            if steps >= 50:
                successes.append(float(steps))
            else:
                failures.append(-float(50 - steps))

        env.close()

        if successes and failures:
            report = evaluate_structural_geometry(np.array(successes), np.array(failures))
            assert isinstance(report, dict)
            assert "ranking_score" in report
            assert "alpha_s_sil" in report
            assert "entropy" in report
            assert "structural_alpha" in report
            assert "success_rate" in report
            assert isinstance(report["ranking_score"], float)
