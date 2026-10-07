# Karakana Mathematical Contract

This document separates the mathematical objects used by Karakana so the
platform does not use the same name for different quantities.

## 1. Moment SGO Matrix

This is the theoretical Structural Geometry Optimization matrix.

For each coherent unit of experience `u` such as an episode, chunk, trade
window, or trajectory segment:

```text
g(r_t) = sign(r_t) * abs(r_t)^p
P_u = sum(max(g(r_t), 0))
N_u = -sum(min(g(r_t), 0))
v_u = [1, P_u, N_u]^T
```

Karakana's production default is `p = 1.1`. This acts as an **Angular Stiffness Warp**: it preserves outcome sign and ordering while magnifying larger magnitudes slightly more than linearly, effectively stretching the Euclidean space to structurally penalize heavy-tailed occurrences. Use `p = 1.0` when a raw theoretical moment matrix is required for derivations or exact hand checks.

The Moment SGO matrix is:

```text
S = E[v_u v_u^T]
```

which expands to:

```text
[
  [1,    E[P],   E[N]],
  [E[P], E[P^2], E[PN]],
  [E[N], E[PN],  E[N^2]],
]
```

`E[PN]` measures within-unit co-occurrence of gains and losses. This requires
per-unit outcomes. It cannot be recovered from separated positive and negative
magnitude arrays after unit structure has been discarded.

Implementation:

- `karakana.metrics.compute_sgo_matrix`
- `karakana.metrics.calculate_moment_sgo_metrics`
- `karakana.trainers.spg.SPGTrainer` when `loss_module="moment"` uses recent
  episode reward sequences as Moment SGO units and honors the configured
  `utility_power`.

Rigor note: for fixed `p`, scaling all rewards by a positive constant scales
`P` and `N` by that constant raised to `p`, so the normalized coupling
`rho_pn_sq = E[PN]^2 / (E[P^2]E[N^2])` remains scale invariant. Raw Frobenius
matrix distance is not fully scale invariant because the augmented constant
dimension remains `1`; use normalized or correlation-style metrics for
cross-scale comparisons.

**Geometric Interpretation (Cosine Inner Product Identity):** The coupling ratio `rho_pn_sq` is mathematically identical to the squared cosine of the angle $\theta$ between the N-dimensional positive event vector $P$ and negative event vector $N$ in $L^2$ space. Specifically, $\rho_{PN} = \cos(\theta) = \frac{\langle P, N \rangle}{\|P\|_2 \|N\|_2}$. Thus, coupling is a strict geometric projection measuring the angular alignment between gains and losses over a unit trajectory, bounded naturally in $[0, 1]$.

The current decoupled ideal preserves the observed first and second marginal
moments while setting `E[PN] = 0`. This isolates coupling without changing the
marginal size of wins or losses.

## 2. Legacy Count/Rate Cofactor Diagnostic

`evaluate_structural_geometry` currently preserves the historical Karakana
reporting behavior. It builds:

```text
N_legacy = [
  [avg_win,   avg_loss,  s],
  [loss_count, win_count, total_count],
  [loss_rate, win_rate, 1],
]
```

This matrix is useful as a compact rate-vs-magnitude diagnostic, but it is not
the Moment SGO matrix. It cannot directly measure `E[PN]` coupling.

Implementation:

- `karakana.metrics.evaluate_structural_geometry`

Migration rule: existing reports and calibration flows may continue using this
legacy diagnostic until their output schemas are versioned. New mathematical
work should prefer the Moment SGO matrix when unit-level outcomes are available.

## 3. Alpha-S-SIL

`alpha_s_sil` is a legacy opposite-sign alignment loss. It compares:

- `ideal_alpha`: the third element of the second row in the ideal cofactor matrix.
- `s_observed`: the third element of the first row in the legacy `n_matrix`.

The intended perfect condition is:

```text
-ideal_alpha = s_observed
```

with `s_observed` positive and `ideal_alpha` negative. Equivalently,
`ideal_alpha + s_observed = 0`. The loss is:

```text
alpha_s_sil = (ideal_alpha + s_observed)^2 /
              (ideal_alpha^2 + s_observed^2 + eps)
```

Therefore lower is better, and zero is perfect opposite-sign alignment.

**Raw Bounds and Trigonometric Identity:**
The mathematical limits of the raw `alpha_s_sil` formulation yield a range of `[0, 2]`. This is not an arbitrary scale; it stems directly from Euclidean geometry in $\mathbb{R}^2$.

**Proposition (Trigonometric Identity of Macro Alignment Loss):**
Let $\mathbf{a} = [\alpha_{\text{ideal}}, s_{\text{observed}}]^\top \in \mathbb{R}^2$ be the 2D macro alignment state vector, where $\alpha_{\text{ideal}} < 0$ denotes the target cofactor expectation and $s_{\text{observed}} = \mu_P - \mu_N$ denotes the observed net utility spread. Let $\mathbf{d} = [1, 1]^\top \in \mathbb{R}^2$ define the same-sign structural collapse direction.

The normalized macro alignment loss $\mathcal{L}_{\text{SGO, SIL}}(\theta)$ is identically equal to the squared cosine of the vector angle $\psi_{\mathbf{a}, \mathbf{d}}$ between $\mathbf{a}$ and $\mathbf{d}$ in Euclidean space $\mathbb{R}^2$:

```text
L_SGO,SIL(theta) = lambda_sgo * cos^2(psi_a,d)
```

where

```text
cos(psi_a,d) = <a, d> / (||a||_2 * ||d||_2)
             = (alpha_ideal * 1 + s_observed * 1) / (sqrt(alpha_ideal^2 + s_observed^2) * sqrt(1^2 + 1^2))
             = (alpha_ideal + s_observed) / sqrt(2 * (alpha_ideal^2 + s_observed^2))
```

Squaring both sides yields:

```text
cos^2(psi_a,d) = (alpha_ideal + s_observed)^2 / (2 * (alpha_ideal^2 + s_observed^2))
               = alpha_s_sil / 2
```

Thus:
1. The division by 2.0 is the squared Euclidean norm of the collapse axis: `||d||_2^2 = 1^2 + 1^2 = 2`.
2. Raw diagnostic `alpha_s_sil = 2 * cos^2(psi_a,d)` naturally spans `[0, 2]`.
3. Structural health is the complementary orthogonal alignment:
   `alpha_health = 1 - cos^2(psi_a,d) = 1 - (alpha_s_sil / 2) in [0, 1]`.

**Corollary (Dual-Cosine Scale-Invariant SGPO Objective):**
Let $\theta_{\mathbf{P}, \mathbf{N}}$ be the inner-product angle between the $m$-dimensional trajectory gain vector $\mathbf{P}$ and loss magnitude vector $\mathbf{N}$ in $\mathbb{R}^m$ (Section 1), and let $\psi_{\mathbf{a}, \mathbf{d}}$ be the macro alignment state angle in $\mathbb{R}^2$. The complete Structural Geometry Policy Optimization (SGPO) objective takes the unified scale-invariant trigonometric form:

```text
L_SGPO(theta) = L_PPO(theta) + lambda_sgo * cos^2(psi_a,d) + lambda_coupling * cos^2(theta_P,N)
```

where:
- `L_PPO(theta) = L_CLIP(theta) + c_1 * L_VF(theta) - c_2 * H[pi_theta]` bounds parameter step size.
- `cos^2(psi_a,d)` measures macro matrix alignment error as squared cosine distance to the diagonal collapse axis `d = [1, 1]^T` in $\mathbb{R}^2$.
- `cos^2(theta_P,N) = rho_pn_sq` measures micro trajectory coupling as squared cosine collinearity between gain and loss vectors in $\mathbb{R}^m$.

## 4. Probability SGO Matrix

The probability matrix discretizes individual outcomes by direction and
magnitude:

```text
direction = loss | neutral | win
magnitude = small | medium | large
```

This is a monitoring and differentiable-training projection, not a replacement
for Moment SGO. Its target and divergence direction must be named explicitly:

- Target-likelihood style KL: penalizes low mass in the target cell.
- Observed-to-target KL: requires a smoothed non-zero target distribution.
- Frobenius distance: direct matrix distance to a target matrix.

Current PyTorch `probability_kl` behaves like target-to-observed KL because of
the PyTorch `kl_div(log_observed, target)` convention.

`karakana.probability.evaluate_probability_geometry` exposes both explicit
directions. The legacy `kl_divergence` field is retained as
observed-to-target KL and is accompanied by `kl_divergence_direction`.
`target_likelihood_kl` is the PyTorch-compatible training/audit loss.

PyTorch implementations must share the same probability matrix and divergence
semantics. `karakana.pytorch.SGOLossModule`, `karakana.torch_loss.TorchSGOLoss`,
and `karakana.torch_metrics` use the same soft probability projection for
`probability_kl` and `probability_fro`. `calculate_rho_pn_sq` applies the same
utility transform as Moment SGO and treats each row of a 2-D tensor as one
outcome unit.

### Coupling Trajectory Extension (`TorchSGOLoss`)

As of the SGPO proposal (`bak/torchloss_improvement.txt`), `TorchSGOLoss`
supports a configurable `coupling_K` trajectory penalty (`use_coupling_trajectory`,
`lambda_coupling`, `coupling_K`, `coupling_gradient`). When enabled, the module
computes rolling chunk-level `ρ²_PN` (chunk size = `coupling_K`, default 10)
directly inside `forward` and adds it to the structural regularizer. The
gradient-decay option (`coupling_gradient`) approximates a trajectory constraint
so that extreme coupling must decrease over training epochs. See
`karakana/torch_loss.py` for the implementation.

## 5. Streaming Alpha

Streaming alpha is an operational health score:

```text
alpha = max(0, 1 - min(1, structural_loss + content_penalty))
```

It is bounded in `[0, 1]`. It is not the Moment SGO matrix, and it is not the
legacy `structural_alpha` cofactor. It is suitable for live telemetry, alerting,
and collapse-horizon comparisons after calibration.

Collapse detection uses the EMA-smoothed `V_alpha`, not the raw one-step alpha
drop. The raw drop is still reported as `max_alpha_drop` for diagnostics.
Horizon units must be explicit: `horizon_chunks` is computed as
`(ema_alpha - safety_floor) / abs(V_alpha)` when `V_alpha < 0`, and
`horizon_tokens = horizon_chunks * chunk_size`. The legacy `horizon` response
field aliases `horizon_tokens`.
