"""Tests for SGOLoss trajectory-aware auto-tuning."""

from karakana.loss import SGOLoss


class TestSGOLoss:
    def test_constructor_defaults(self):
        loss = SGOLoss()
        assert loss.base_lambda == 0.1
        assert loss.current_lambda == 0.1
        assert loss.auto_tune is True
        assert loss.metric_key == "alpha_s_sil"
        assert loss.utility_power == 1.1

    def test_add_outcome_and_regularization(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=False)
        loss.add_outcome(10.0)
        loss.add_outcome(-10.0)
        reg = loss.calculate_regularization()
        assert isinstance(reg, float)
        assert 0.0 <= reg <= 1.0

    def test_regularization_returns_zero_with_no_outcomes(self):
        loss = SGOLoss(auto_tune=False)
        assert loss.calculate_regularization() == 0.0

    def test_regularization_returns_penalty_with_single_sign(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=False)
        loss.add_outcome(5.0)
        loss.add_outcome(3.0)
        reg = loss.calculate_regularization()
        assert reg == 0.1

    def test_bad_outcomes_produce_higher_penalty_than_good(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=False)

        loss.add_outcome(10.0)
        loss.add_outcome(-20.0)
        bad_reg = loss.calculate_regularization()
        loss.reset()

        loss.add_outcome(15.0)
        loss.add_outcome(-2.0)
        good_reg = loss.calculate_regularization()

        assert bad_reg >= good_reg

    def test_auto_tune_tightens_lambda_on_bad_outcomes(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=True, trajectory_window_step=1)
        initial_lambda = loss.current_lambda
        for _ in range(60):
            loss.add_outcome(10.0)
            loss.add_outcome(-20.0)
            loss.calculate_regularization()
        assert loss.current_lambda > initial_lambda

    def test_auto_tune_relaxes_lambda_on_good_outcomes(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=True)
        loss.current_lambda = 0.5
        for _ in range(15):
            loss.add_outcome(15.0)
            loss.add_outcome(-2.0)
            loss.calculate_regularization()
            loss.reset()
        assert loss.current_lambda < 0.5

    def test_reset_clears_state(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=True)
        loss.add_outcome(10.0)
        loss.reset()
        assert len(loss.outcomes) == 0
        assert loss._step_counter == 0
        assert loss.current_lambda == loss.base_lambda

    def test_get_tracker_stats(self):
        loss = SGOLoss(auto_tune=False)
        stats = loss.get_tracker_stats()
        assert "alpha" in stats
        assert "v_alpha" in stats
        assert "m_alpha" in stats
        assert isinstance(stats["alpha"], float)

    def test_total_loss(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=False)
        loss.add_outcome(10.0)
        loss.add_outcome(-10.0)
        total = loss.total_loss(0.5)
        assert isinstance(total, float)
        assert total >= 0.5

    def test_single_outcome_side_behavior(self):
        loss = SGOLoss(base_lambda=0.1, auto_tune=False)
        for _ in range(5):
            loss.add_outcome(5.0)
        reg = loss.calculate_regularization()
        assert reg == 0.1
