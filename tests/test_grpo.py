import pytest
import torch
from karakana_engine.trainers.grpo import GRPOTrainer, GRPORefPolicy

def test_grpo_init_and_step():
    trainer = GRPOTrainer(
        "CartPole-v1",
        group_size=2,
        episodes=1,
        lr=1e-3,
        lambda_sgo=0.1,
    )
    assert isinstance(trainer.policy, GRPORefPolicy)
    assert isinstance(trainer.ref_policy, GRPORefPolicy)

    reward, loss = trainer.train_episode()
    assert isinstance(reward, (float, int))
    assert isinstance(loss, (float, int))
    assert 0.0 <= trainer.current_alpha <= 1.0

def test_grpo_policy_state_dict():
    policy = GRPORefPolicy(obs_size=4, action_count=2)
    state_dict = policy.state_dict()
    assert any("net." in k for k in state_dict.keys())
    assert not any("actor." in k for k in state_dict.keys())
    assert not any("critic." in k for k in state_dict.keys())
