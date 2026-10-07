import pytest
from karakana_api.training_runtime import should_save_best_checkpoint


def test_should_save_best_checkpoint_initial():
    should_save, reason = should_save_best_checkpoint(
        reward=100.0,
        alpha=0.6,
        best_reward=-float("inf"),
        best_alpha=-float("inf"),
    )
    assert should_save is True
    assert reason == "initial"


def test_should_save_best_checkpoint_decisive_breakthrough():
    # best is 100.0, tol is max(100 * 0.03, 0.05) = 3.0
    # reward 105.0 is diff +5.0 > 3.0 -> breakthrough saved even if alpha is lower
    should_save, reason = should_save_best_checkpoint(
        reward=105.0,
        alpha=0.4,
        best_reward=100.0,
        best_alpha=0.8,
    )
    assert should_save is True
    assert "reward breakthrough" in reason


def test_should_save_best_checkpoint_near_equal_higher_alpha():
    # best is 100.0, best_alpha is 0.50
    # candidate has reward 99.0 (within 3% tol of 100), but alpha is 0.75
    # should save for structural stability!
    should_save, reason = should_save_best_checkpoint(
        reward=99.0,
        alpha=0.75,
        best_reward=100.0,
        best_alpha=0.50,
    )
    assert should_save is True
    assert "structural stability" in reason


def test_should_save_best_checkpoint_near_equal_lower_alpha():
    # best is 100.0, best_alpha is 0.80
    # candidate has slightly higher reward 101.0 (diff +1.0 <= tol 3.0), but lower alpha 0.50
    # fragile policy should NOT overwrite robust policy
    should_save, reason = should_save_best_checkpoint(
        reward=101.0,
        alpha=0.50,
        best_reward=100.0,
        best_alpha=0.80,
    )
    assert should_save is False
    assert reason == "inferior"


def test_should_save_best_checkpoint_near_equal_equal_alpha():
    # best is 100.0, best_alpha is 0.70
    # candidate has reward 101.5 (diff +1.5 <= tol 3.0), identical alpha 0.702 (within alpha_delta 0.005)
    # slight reward gain accepted
    should_save, reason = should_save_best_checkpoint(
        reward=101.5,
        alpha=0.702,
        best_reward=100.0,
        best_alpha=0.70,
    )
    assert should_save is True
    assert "slight reward gain" in reason


def test_should_save_best_checkpoint_negative_rewards():
    # MountainCar domain: rewards are -200, -180, etc.
    # best is -200.0, tol is max(abs(-200) * 0.03, 0.05) = 6.0
    # candidate is -198.0 (diff +2.0 <= tol 6.0), but alpha is 0.65 vs 0.50
    should_save, reason = should_save_best_checkpoint(
        reward=-198.0,
        alpha=0.65,
        best_reward=-200.0,
        best_alpha=0.50,
    )
    assert should_save is True
    assert "structural stability" in reason

    # Breakthrough in negative reward: -180.0 (diff +20.0 > 6.0)
    should_save, reason = should_save_best_checkpoint(
        reward=-180.0,
        alpha=0.45,
        best_reward=-200.0,
        best_alpha=0.65,
    )
    assert should_save is True
    assert "reward breakthrough" in reason
