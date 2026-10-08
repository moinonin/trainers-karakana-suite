import base64
import io
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split

from karakana_engine.pytorch import TorchSGOLoss
from karakana_engine.trainers.grg import GRGController

PREDICTION_OUTPUT_COLUMNS = {
    "confidence",
    "is_ai",
    "predicted_class",
    "predicted_label",
    "predicted_probability",
    "prediction",
    "probability",
}


def is_exported_index_column(column: str) -> bool:
    """Return whether a CSV header is a conventional serialized DataFrame index."""
    normalized = column.strip().lower()
    return normalized == "unnamed" or normalized.startswith("unnamed:")


class DatasetPolicy(nn.Module):
    """Simple MLP for dataset-backed supervised learning."""

    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class DatasetTrainer:
    """
    SGO-regularized trainer for static datasets.
    Supports classification (CrossEntropy + SGO) and regression (MSE + SGO).
    """

    def __init__(
        self,
        dataset_csv_b64: str,
        target_col: str,
        task_type: str = "classification",
        lambda_sgo: float = 0.1,
        lr: float = 0.001,
        loss_module: str = "moment",
        ranking_profile: str = "conservative",
        steering_enabled: bool = True,
        random_seed: int = 42,
        lambda_coupling=None,
        coupling_k=None,
        coupling_gradient=None,
        use_coupling_trajectory=None,
        utility_power=None,
        num_epochs=None,
    ):
        # 1. Load Data
        csv_data = base64.b64decode(dataset_csv_b64)
        self.df = pd.read_csv(io.BytesIO(csv_data))

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        if self.df.empty:
            raise ValueError("Dataset must contain at least one row")
        if self.df[target_col].isna().any():
            raise ValueError(f"Target column '{target_col}' contains missing values")

        self.target_col = target_col
        self.task_type = task_type
        self.random_seed = random_seed
        self.y_raw = self.df[target_col].to_numpy()
        self.feature_columns, self.ignored_columns = self._select_feature_columns()
        if not self.feature_columns:
            raise ValueError(
                "Dataset has no usable numeric or boolean feature columns after excluding "
                f"the target and non-feature columns: {self.ignored_columns}"
            )

        feature_frame = self.df[self.feature_columns].copy()
        for column in self.feature_columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)

        all_missing = [column for column in feature_frame if feature_frame[column].isna().all()]
        if all_missing:
            self.feature_columns = [column for column in self.feature_columns if column not in all_missing]
            self.ignored_columns.extend(all_missing)
            feature_frame = feature_frame.drop(columns=all_missing)
        if feature_frame.shape[1] == 0:
            raise ValueError("All candidate feature columns are empty or non-finite")

        self.ranking_profile = ranking_profile

        # Exact duplicate rows stay in one partition by splitting unique rows.
        unique_frame = pd.concat([feature_frame, self.df[[target_col]]], axis=1).drop_duplicates()
        if len(unique_frame) < 7:
            raise ValueError("Dataset requires at least seven distinct rows for train/validation/test splitting")
        feature_frame = unique_frame[self.feature_columns]
        target_values = unique_frame[target_col].to_numpy()
        row_indices = np.arange(len(unique_frame))
        stratify = target_values if task_type == "classification" else None
        try:
            train_indices, remainder_indices = train_test_split(
                row_indices,
                test_size=0.30,
                random_state=random_seed,
                stratify=stratify,
            )
            remainder_targets = target_values[remainder_indices]
            validation_indices, test_indices = train_test_split(
                remainder_indices,
                test_size=0.50,
                random_state=random_seed,
                stratify=remainder_targets if task_type == "classification" else None,
            )
        except ValueError as exc:
            raise ValueError(f"Dataset cannot form a reproducible 70/15/15 split: {exc}") from exc

        train_features = feature_frame.iloc[train_indices]
        self.feature_medians = train_features.median(numeric_only=True)
        self.feature_means = train_features.fillna(self.feature_medians).mean()
        self.feature_scales = train_features.fillna(self.feature_medians).std(ddof=0).replace(0.0, 1.0)

        def normalize(indices: np.ndarray) -> np.ndarray:
            selected = feature_frame.iloc[indices].fillna(self.feature_medians)
            return ((selected - self.feature_means) / self.feature_scales).to_numpy(dtype=np.float32)

        # 2. Setup Model
        input_dim = len(self.feature_columns)
        if task_type == "classification":
            self.unique_classes = np.unique(target_values)
            if len(self.unique_classes) < 2:
                raise ValueError("Classification requires at least two target classes")
            output_dim = len(self.unique_classes)
            self.class_map = {val: i for i, val in enumerate(self.unique_classes)}
            encoded = np.array([self.class_map[v] for v in target_values], dtype=np.int64)
            self.base_loss_fn = nn.CrossEntropyLoss()
            target_tensor = torch.tensor(encoded, dtype=torch.long)
            self.class_labels = [
                value.item() if isinstance(value, np.generic) else value for value in self.unique_classes
            ]
            self.target_mean = None
            self.target_scale = None
        else:
            output_dim = 1
            numeric_target = pd.to_numeric(pd.Series(target_values), errors="coerce")
            if not np.isfinite(numeric_target.to_numpy(dtype=float)).all():
                raise ValueError(f"Regression target column '{target_col}' must contain only finite numeric values")
            numeric_values = numeric_target.to_numpy(dtype=np.float32)
            self.target_mean = float(np.mean(numeric_values[train_indices]))
            self.target_scale = float(np.std(numeric_values[train_indices]))
            if self.target_scale <= 1e-12:
                self.target_scale = 1.0
            normalized_target = (numeric_values - self.target_mean) / self.target_scale
            target_tensor = torch.tensor(normalized_target, dtype=torch.float32).view(-1, 1)
            self.base_loss_fn = nn.MSELoss()
            self.class_map = {}
            self.class_labels = []

        self.output_dim = output_dim
        self.X = torch.from_numpy(normalize(train_indices))
        self.y = target_tensor[train_indices]
        self.validation_X = torch.from_numpy(normalize(validation_indices))
        self.validation_y = target_tensor[validation_indices]
        self.test_X = torch.from_numpy(normalize(test_indices))
        self.test_y = target_tensor[test_indices]
        self._sampling_generator = torch.Generator().manual_seed(random_seed)
        self._sample_weights: torch.Tensor | None = None
        self.training_class_counts: dict[str, int] = {}
        self.sampling_strategy = "seeded_shuffle"
        if task_type == "classification":
            train_class_counts = torch.bincount(self.y, minlength=output_dim)
            self.training_class_counts = {
                str(self.class_labels[index]): int(count) for index, count in enumerate(train_class_counts.tolist())
            }
            inverse_counts = train_class_counts.to(torch.float64).clamp_min(1).reciprocal()
            self._sample_weights = inverse_counts[self.y]
            self.sampling_strategy = "inverse_frequency_oversampling"
        self._split_sizes = {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        }
        self.selection_metric_name = "balanced_accuracy" if task_type == "classification" else "negative_rmse"
        with torch.random.fork_rng():
            torch.manual_seed(random_seed)
            self.policy = DatasetPolicy(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 3. SGO Setup
        self.sgo_loss_fn = TorchSGOLoss(
            lambda_sgo=lambda_sgo,
            metric_key="ranking_score" if loss_module == "moment" else "alpha_s_sil",
            loss_module=loss_module,
            ranking_profile=ranking_profile,
            lambda_coupling=lambda_coupling if lambda_coupling is not None else 0.0,
            coupling_k=coupling_k if coupling_k is not None else 10,
            coupling_gradient=coupling_gradient if coupling_gradient is not None else False,
            use_coupling_trajectory=use_coupling_trajectory if use_coupling_trajectory is not None else False,
            utility_power=utility_power if utility_power is not None else 1.1,
        )
        self.grg = GRGController(self.optimizer, base_lr=lr)
        self.steering_enabled = steering_enabled
        self._last_alpha = 1.0

    def _select_feature_columns(self) -> tuple[list[str], list[str]]:
        features: list[str] = []
        ignored: list[str] = []
        for column in self.df.columns:
            if column == self.target_col:
                continue
            normalized_name = column.strip().lower()
            if is_exported_index_column(column):
                ignored.append(column)
                continue
            if normalized_name in PREDICTION_OUTPUT_COLUMNS:
                ignored.append(column)
                continue
            series = self.df[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                if self.task_type == "classification" and self._is_target_proxy(series):
                    ignored.append(column)
                    continue
                features.append(column)
            else:
                ignored.append(column)
        return features, ignored

    def _is_target_proxy(self, feature: pd.Series) -> bool:
        target = self.df[self.target_col]
        pair = pd.DataFrame({"feature": feature, "target": target}).dropna()
        feature_classes = pair["feature"].nunique()
        target_classes = pair["target"].nunique()
        if feature_classes < 2 or feature_classes != target_classes or feature_classes > 20:
            return False
        feature_to_target = pair.groupby("feature", dropna=False)["target"].nunique().max()
        target_to_feature = pair.groupby("target", dropna=False)["feature"].nunique().max()
        return int(feature_to_target) == 1 and int(target_to_feature) == 1

    def dataset_summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.df)),
            "target_column": self.target_col,
            "task_type": self.task_type,
            "ranking_profile": self.ranking_profile,
            "feature_count": len(self.feature_columns),
            "feature_columns": list(self.feature_columns),
            "ignored_columns": sorted(set(self.ignored_columns)),
            "class_count": int(len(self.unique_classes)) if self.task_type == "classification" else None,
            "training_class_counts": self.training_class_counts,
            "sampling_strategy": self.sampling_strategy,
            "split_sizes": self.split_sizes(),
            "random_seed": self.random_seed,
        }

    def split_sizes(self) -> dict[str, int]:
        return dict(self._split_sizes)

    def preprocessing_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "feature_columns": list(self.feature_columns),
            "expected_types": {column: "number_or_boolean" for column in self.feature_columns},
            "medians": {column: float(self.feature_medians[column]) for column in self.feature_columns},
            "means": {column: float(self.feature_means[column]) for column in self.feature_columns},
            "scales": {column: float(self.feature_scales[column]) for column in self.feature_columns},
            "target_transform": (
                {
                    "method": "standard",
                    "mean": self.target_mean,
                    "scale": self.target_scale,
                }
                if self.task_type == "regression"
                else {"method": "none"}
            ),
            "missing_value_policy": "replace null with the training-split median",
        }

    def _evaluate(self, features: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
        self.policy.eval()
        with torch.inference_mode():
            outputs = self.policy(features)
        self.policy.train()
        if self.task_type == "classification":
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            actual = targets.cpu().numpy()
            precision, recall, _, _ = precision_recall_fscore_support(
                actual,
                predicted,
                labels=np.arange(len(self.class_labels)),
                zero_division=0,
            )
            return {
                "accuracy": float(accuracy_score(actual, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
                "per_class": {
                    str(label): {"precision": float(precision[index]), "recall": float(recall[index])}
                    for index, label in enumerate(self.class_labels)
                },
                "confusion_matrix": confusion_matrix(
                    actual,
                    predicted,
                    labels=np.arange(len(self.class_labels)),
                ).tolist(),
                "mean_confidence": float(probabilities.max(axis=1).mean()),
            }
        predicted = outputs.view(-1).cpu().numpy() * self.target_scale + self.target_mean
        actual = targets.view(-1).cpu().numpy() * self.target_scale + self.target_mean
        residuals = predicted - actual
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r_squared": float(r2_score(actual, predicted)) if len(actual) > 1 else 0.0,
            "residual_percentiles": {
                "p05": float(np.percentile(residuals, 5)),
                "p50": float(np.percentile(residuals, 50)),
                "p95": float(np.percentile(residuals, 95)),
            },
            "target_range": [float(np.min(actual)), float(np.max(actual))],
            "predicted_range": [float(np.min(predicted)), float(np.max(predicted))],
        }

    def validation_metrics(self) -> dict[str, Any]:
        return self._evaluate(self.validation_X, self.validation_y)

    def validation_score(self) -> float:
        metrics = self.validation_metrics()
        if self.task_type == "classification":
            return float(metrics["balanced_accuracy"])
        return -float(metrics["rmse"])

    def test_metrics(self) -> dict[str, Any]:
        return self._evaluate(self.test_X, self.test_y)

    def model_card(self, job: Any, metrics: dict[str, Any]) -> str:
        return (
            f"# Karakana Dataset Model\n\n"
            f"- Job ID: `{job.job_id}`\n"
            f"- Task: `{self.task_type}`\n"
            f"- Target: `{self.target_col}`\n"
            f"- Features: `{', '.join(self.feature_columns)}`\n"
            f"- Split sizes: `{self.split_sizes()}`\n"
            f"- Ranking profile: `{self.ranking_profile}`\n"
            f"- Test metrics: `{metrics['test']}`\n\n"
            "## Intended Use\nUse the authenticated Karakana prediction endpoint with records matching the feature schema.\n\n"
            "## Limitations\nMetrics are specific to the uploaded dataset and split seed. They do not establish production readiness or future performance.\n"
        )

    @property
    def current_alpha(self):
        return self._last_alpha

    def train_epoch(self, batch_size=32):
        """Run one training epoch across the dataset."""
        if self._sample_weights is not None:
            indices = torch.multinomial(
                self._sample_weights,
                num_samples=self.X.size(0),
                replacement=True,
                generator=self._sampling_generator,
            )
        else:
            indices = torch.randperm(
                self.X.size(0),
                generator=self._sampling_generator,
            )
        epoch_loss = 0.0

        for i in range(0, self.X.size(0), batch_size):
            idx = indices[i : i + batch_size]
            batch_X, batch_y = self.X[idx], self.y[idx]

            self.optimizer.zero_grad()
            outputs = self.policy(batch_X)

            # Base Task Loss
            base_loss = self.base_loss_fn(outputs, batch_y)

            # SGO Loss (Structural regularizer)
            if self.task_type == "classification":
                # Positive means the true class beats every alternative; negative
                # means the model structurally preferred an incorrect class.
                true_logits = outputs.gather(1, batch_y.unsqueeze(1)).squeeze(1)
                competing_logits = outputs.clone()
                competing_logits.scatter_(1, batch_y.unsqueeze(1), float("-inf"))
                scores = true_logits - competing_logits.max(dim=1).values
            else:
                # Center errors around the detached batch baseline so better-than-
                # baseline predictions are positive and worse predictions negative.
                absolute_error = torch.abs(outputs - batch_y).view(-1)
                scores = absolute_error.mean().detach() - absolute_error

            total_loss = self.sgo_loss_fn(scores, base_loss=base_loss)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.item()

            # Extract alpha for steering
            with torch.no_grad():
                # Raw structural term from SGO loss
                from karakana_engine.metrics import evaluate_structural_geometry

                pos = scores[scores > 0].cpu().numpy()
                neg = scores[scores <= 0].cpu().numpy()
                if pos.size > 0 and neg.size > 0:
                    res = evaluate_structural_geometry(pos, neg)
                    self._last_alpha = max(
                        0.0,
                        min(1.0, 1.0 - float(res.get("alpha_s_sil", 1.0)) / 2.0),
                    )

            if self.steering_enabled:
                self.grg.step(self._last_alpha)

        batch_count = max(1, (self.X.size(0) + batch_size - 1) // batch_size)
        avg_loss = epoch_loss / batch_count
        return avg_loss, self._last_alpha

    def train(self, epochs=10):
        metrics = []
        for epoch in range(1, epochs + 1):
            loss, alpha = self.train_epoch()
            metrics.append({"epoch": epoch, "loss": loss, "alpha": alpha})
        return metrics
