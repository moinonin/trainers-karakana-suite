"""Centralized configuration loader for karakana."""

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/karakana-config.json")


def resolve_config_path() -> Path:
    """Find the config file regardless of current working directory."""
    env_path = os.getenv("KARAKANA_CONFIG")
    if env_path:
        return Path(env_path)

    # 1. Try relative to CWD
    cwd_path = Path.cwd() / DEFAULT_CONFIG_PATH
    if cwd_path.exists():
        return cwd_path

    # 2. Try relative to this file (karakana/config.py -> ../config/...)
    module_path = Path(__file__).parent.parent / DEFAULT_CONFIG_PATH
    if module_path.exists():
        return module_path

    return DEFAULT_CONFIG_PATH


def load_config() -> dict[str, Any]:
    """Load the karakana configuration from file or return defaults."""
    # Global Defaults
    config = {
        "global": {
            "ranking_profile": "balanced",
            "loss_module": "probability_kl",
            "lambda_sgo": 0.1,
            "coupling_K": 10,
            "use_coupling_trajectory": True,
            "lambda_coupling": 0.5,
            "coupling_gradient": False,
            "entropy_coef": 0.0,
            "adaptive_lambda": False,
            "cloud_sync": False,
            "sgo_buffer_size": 20,
            "v_alpha_panic_threshold": -0.01,
            "alpha_safety_floor": 0.40,
        },
        "profiles": {},
    }

    config_path = resolve_config_path()

    if config_path.exists():
        try:
            with open(config_path) as f:
                user_config = json.load(f)
                if "global" in user_config or "profiles" in user_config:
                    # New hierarchical format
                    if "global" in user_config:
                        config["global"].update(user_config["global"])
                    if "profiles" in user_config:
                        config["profiles"].update(user_config["profiles"])
                else:
                    # Legacy flat format - merge into global
                    config["global"].update(user_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    return config


# Global config instance
_config_data = load_config()


class ConfigWrapper:
    """Helper to access config with legacy support and profile awareness."""

    def __init__(self, data):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Legacy access to global settings."""
        return self._data["global"].get(key, default)

    def get_profile(self, profile_name: str) -> dict[str, Any]:
        """Get profile merged with global defaults."""
        profile = self._data["global"].copy()
        profile.update(self._data["profiles"].get(profile_name, {}))
        return profile

    def get_profile_grg(self, profile_name: str = None) -> dict[str, Any]:
        """Get GRG settings merged: global.grg → profile.grg."""
        grg = dict(self._data["global"].get("grg", {}))
        if profile_name:
            profile_grg = self._data["profiles"].get(profile_name, {}).get("grg", {})
            grg.update(profile_grg)
        return grg

    def get_gym_env_profile_name(self, env_name: str | None) -> str:
        """Return the canonical profile that owns an RL environment."""
        if env_name == "EvxHistoricalTrading-v0":
            return "evx_historical_trading"
        if env_name in {
            "SGO-Trading-v0",
            "HistoricalTrading-v0",
        }:
            return "hft-profile"
        if env_name == "HistoricalTrading-v1":
            return "HistoricalTrading-v1"
        return "gym_rl"

    def get_gym_env_defaults_key(self, env_name: str) -> str:
        """Return the gym_envs key used for env-specific defaults."""
        if env_name and "NoFrameskip" in env_name:
            return "Atari-Pixel-v4"
        return env_name

    def get_gym_env_defaults(self, env_name: str) -> dict[str, Any]:
        """Resolve trainer defaults for one Gym environment.

        Env-specific settings override their owning profile, while GRG
        thresholds always come from the profile's canonical ``grg`` block.
        """
        profile_name = self.get_gym_env_profile_name(env_name)
        profile = self.get_profile(profile_name)
        gym_profile = self.get_profile("gym_rl")
        env_defaults_key = self.get_gym_env_defaults_key(env_name)
        env_defaults = dict((gym_profile.get("gym_envs") or {}).get(env_defaults_key, {}))
        grg = self.get_profile_grg(profile_name)

        resolved = {
            "trainer_type": env_defaults.get("trainer_type", profile.get("trainer_type", "spg")),
            "grpo_group_size": env_defaults.get("grpo_group_size", profile.get("grpo_group_size", 4)),
            "grpo_clip_eps": env_defaults.get("grpo_clip_eps", profile.get("grpo_clip_eps", 0.2)),
            "grpo_kl_coef": env_defaults.get("grpo_kl_coef", profile.get("grpo_kl_coef", 0.04)),
            "lambda_sgo": env_defaults.get("lambda_sgo", profile.get("lambda_sgo", 0.05)),
            "learning_rate": env_defaults.get("learning_rate", profile.get("learning_rate", 0.01)),
            "loss_module": env_defaults.get("loss_module", profile.get("loss_module", "moment")),
            "ranking_profile": env_defaults.get("ranking_profile", profile.get("ranking_profile", "conservative")),
            "episodes": env_defaults.get("episodes", profile.get("episodes", 200)),
            "entropy_coef": env_defaults.get("entropy_coef", profile.get("entropy_coef", 0.0)),
            "tail_risk_coef": env_defaults.get("tail_risk_coef", profile.get("tail_risk_coef", 0.0)),
            "batch_size": env_defaults.get("batch_size", profile.get("batch_size", 512)),
            "updates_per_epoch": env_defaults.get(
                "updates_per_epoch",
                profile.get("updates_per_epoch", 100),
            ),
            "gamma": env_defaults.get("gamma", profile.get("gamma", 0.99)),
            "conservative_weight": env_defaults.get(
                "conservative_weight",
                profile.get("conservative_weight", 1.0),
            ),
            "early_stopping_patience": env_defaults.get(
                "early_stopping_patience",
                profile.get("early_stopping_patience"),
            ),
            "early_stopping_min_delta": env_defaults.get(
                "early_stopping_min_delta",
                profile.get("early_stopping_min_delta", 1e-4),
            ),
            "max_rows": env_defaults.get("max_rows", profile.get("max_rows")),
            "random_seed": env_defaults.get(
                "random_seed",
                profile.get("random_seed", 42),
            ),
            "dataset_path": env_defaults.get(
                "dataset_path",
                profile.get("dataset_path"),
            ),
            "dataset_sha256": env_defaults.get(
                "dataset_sha256",
                profile.get("dataset_sha256"),
            ),
            "evx_shifts": env_defaults.get("evx_shifts", [1, 5, 20]),
            "evx_label_threshold": env_defaults.get("evx_label_threshold", 0.0025),
            "train_fraction": env_defaults.get("train_fraction", 0.70),
            "validation_fraction": env_defaults.get("validation_fraction", 0.15),
            "adaptive_lambda": env_defaults.get("adaptive_lambda", profile.get("adaptive_lambda", False)),
            "sgo_buffer_size": env_defaults.get(
                "sgo_buffer_size", profile.get("sgo_buffer_size", self.get("sgo_buffer_size", 20))
            ),
            "grg_profile": profile_name,
            "grg_window_size": grg.get("window_size", self.get("sgo_buffer_size", 20)),
            "alpha_safety_floor": grg.get("alpha_safety_floor", self.get("alpha_safety_floor", 0.1)),
            "v_alpha_panic_threshold": grg.get("v_alpha_panic_threshold", self.get("v_alpha_panic_threshold", -0.02)),
            "lambda_coupling": env_defaults.get(
                "lambda_coupling", profile.get("lambda_coupling", self.get("lambda_coupling", 0.0))
            ),
            "coupling_k": env_defaults.get("coupling_k", profile.get("coupling_k", self.get("coupling_k", 10))),
            "coupling_gradient": env_defaults.get(
                "coupling_gradient", profile.get("coupling_gradient", self.get("coupling_gradient", False))
            ),
            "use_coupling_trajectory": env_defaults.get(
                "use_coupling_trajectory",
                profile.get("use_coupling_trajectory", self.get("use_coupling_trajectory", False)),
            ),
            "utility_power": env_defaults.get(
                "utility_power", profile.get("utility_power", self.get("utility_power", 1.1))
            ),
            "num_epochs": env_defaults.get("num_epochs", profile.get("num_epochs", self.get("num_epochs", 50))),
            "steering_enabled": env_defaults.get(
                "steering_enabled", profile.get("steering_enabled", self.get("steering_enabled", True))
            ),
            "use_reward_shaping": env_defaults.get(
                "use_reward_shaping", profile.get("use_reward_shaping", self.get("use_reward_shaping", False))
            ),
        }
        return resolved

    def __getitem__(self, key):
        return self._data["global"][key]


get_config = ConfigWrapper(_config_data)
