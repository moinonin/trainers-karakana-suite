from karakana_engine.config import get_config
from karakana_engine.trainers.spg import SPGTrainer


def test_cartpole_config_defaults():
    defaults = get_config.get_gym_env_defaults("CartPole-v1")
    assert defaults["gamma"] == 0.99
    assert defaults["lambda_coupling"] == 0.003
    assert defaults["use_coupling_trajectory"] is True


def test_cartpole_spg_alpha_not_stuck_at_half():
    trainer = SPGTrainer(
        "CartPole-v1",
        lambda_sgo=0.05,
        lr=0.01,
        loss_module="moment",
        ranking_profile="balanced",
        gamma=0.99,
        entropy_coef=0.05,
        lambda_coupling=0.003,
        coupling_k=10,
        use_coupling_trajectory=True,
        cloud_sync=False,
    )

    alphas = []
    for _ in range(5):
        reward, loss = trainer.train_episode()
        alphas.append(trainer.current_alpha)

    # Verify alpha dynamically evaluates from geometric separability rather than constant 0.5
    assert any(abs(alpha - 0.5) > 0.005 for alpha in alphas), f"Alpha stuck at 0.5: {alphas}"
