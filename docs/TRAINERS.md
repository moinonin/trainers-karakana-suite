# Karakana Trainer Guide

Karakana trainers are standalone PyTorch modules. You can use any of them without the Karakana platform, database, Docker, or hosted sidecar services.

Install `torch`, `gymnasium`, `numpy` (and optionally `stable_baselines3` for the callback paths). That's it. Deploy anywhere the Python interpreter runs.

---

## Algorithmic Taxonomy & Foundations

Karakana provides four distinct RL and optimization algorithmic paradigms. Understanding how they differ in network topology, advantage computation, and stability guarantees ensures you select the optimal trainer for your problem:

| Algorithm Family | Karakana Trainers | Network Topology | Advantage Computation ($A_t$) | Primary Stability Mechanism | Best Suited For |
|---|---|---|---|---|---|
| **Actor-Critic (REINFORCE with Baseline)** | `SPGTrainer`, `ContinuousSPGTrainer` | Dual-head: Actor ($\pi_\theta$) + Critic ($V_\phi$) | $A_t = R_t - V_\phi(s_t)$ (trajectory returns minus baseline) | Baseline variance reduction + SGO spectral penalty | Simple discrete/continuous control with dense rewards (CartPole, Pendulum) |
| **Proximal Policy Optimization (PPO)** | `SGPOTrainer`, `ContinuousSGPOTrainer`, `AtariSGPOTrainer` | Dual-head: Actor ($\pi_\theta$) + Critic ($V_\phi$) | Generalized Advantage Estimation (GAE) / normalized returns | Ratio clipping $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ bounding policy shifts | Complex environments, robotic control (MuJoCo), pixel games (Atari) |
| **Group Relative Policy Optimization (GRPO)** | `GRPOTrainer` | **Policy only** ($\pi_\theta$ + frozen $\pi_{\text{ref}}$) — **No Critic Network** | Group-relative normalization: $A_i = \frac{R_i - \bar{R}_G}{\sigma_G + \epsilon}$ | Peer-cohort normalization + KL penalty $D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ | Sparse/delayed rewards, deceptive landscapes (MountainCar), LLM reasoning |
| **Conservative Q-Learning (CQL)** | `OfflineTradingTrainer`, `OfflineTradingTrainerV2` | Q-network $Q_\theta(s, a)$ | Temporal Difference (TD) error: $R + \gamma \max Q - Q$ | Conservative out-of-distribution action penalty + SGO tail-risk loss | Static recorded trade logs, market order books, non-interactive datasets |

### How SGO Couples With Each Family

In traditional RL, the loss objective optimizes reward alone ($L_{\text{RL}}$). In Karakana, every trainer augments the task loss with differentiable Structural Geometry Optimization ($L_{\text{SGO}}$):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{SGO}} \cdot \mathcal{L}_{\text{SGO}}$$

- **In Actor-Critic & PPO:** SGO constrains the actor's trajectory representations, penalizing fragile or erratic policy divergence before rewards collapse.
- **In GRPO:** Because GRPO already samples groups of rollouts for relative ranking, SGO directly audits the structural co-factor geometry across the rollout cohort, making GRPO structurally harmonious with SGO.
- **In CQL:** SGO acts as a structural drawdown and tail-risk constraint, penalizing asymmetric risk exposures in offline execution records.

---

## Quick Reference

| Trainer | File | What it's best for | Key differentiator |
|---------|-----|--------------------|------------------|
| **SPGTrainer** | `spg.py` | Classic Gym control (CartPole, Acrobot) | Simple REINFORCE — rollout-able without state normalization |
| **SGPOTrainer** | `sgpo.py` | Complex Gym control (LunarLander) | PPO clipping using ε=0.20 for bounded updates |
| **GRPOTrainer** | `grpo.py` | Deceptive or sparse environments (MountainCar) | Critic-free group relative advantage + KL reference penalty |
| **ContinuousSPGTrainer** | `continuous_spg.py` | Bounded continuous actions (Pendulum) | Gaussian policy network (mean/std heads) |
| **ContinuousSGPOTrainer** | `continuous_sgpo.py` | MuJoCo robotics (Ant-v5, HalfCheetah, Humanoid) | PPO + RunningMeanStd observation normalization |
| **AtariSGPOTrainer** | `atari_trainer.py` | Physics-based pixel games (Pong, Breakout) | NatureCNN (3 Conv layers + 512 dense) + PPO |
| **DatasetTrainer** | `dataset.py` | Tabular data (CSV files) — no Gym env | Direct module training on (state, action, reward, done) records |
| **EVXTrainer** | `evx.py` | Temporal financial sequence auditing | Sliding window label thresholds with structural tracking |
| **OfflineTradingTrainer** | `offline_trading.py` | Logged trade executions (serialized CSV) | Conservative Q-learning + tail-risk downside penalty |
| **OfflineTradingTrainerV2** | `offline_trading_v2.py` | Order-book microstructure with bid/ask spreads | 4-D feature preprocessing: [mid_price, spread, sma_compare, is_short] |
| **SB3 Structural Callback** | `sb3.py` | Existing Stable-Baselines3 workflows | Wrap `learn()` callback for real-time SGO telemetry |
| **GRGController** | `grg.py` | Custom PyTorch training loops | Adaptive LR decay + E-stop via Alpha-Drift Velocity ($V_\alpha$) |

---

## Getting Started

To install and run locally:

```bash
pip install torch numpy gymnasium # trainer base requirements
```

Optional callbacks/backends:

```bash
# Install SB3 for tooling not built into Stable Baselines
pip install stable-baselines3
```

---

## Core Concepts

### 1. Structural Loss Module (`TorchSGOLoss`)

All trainers apply a differentiable penalty that shapes how the policy gradient explores local peaks before dropping into conservative territory. We specifically avoid hard penalties because they don't collapse resiliency — instead, the metric evaluates divergence directly between sequential plans through spectral similarity within strings.

The penalty varies by granularity type of correlation. A typical invocation style integrates this because we don't expect two equally good solutions to diverge during equilibrium tuning.

### 2. Alpha Velocity Tracker (GRG)

An AlphaMomentum tracker keeps a rolling window of alpha across steps. It detects collapse when the slope exceeds your configured panic threshold — then the GRG controller fires an E-stop as a fallback, restoring parameters on recovery.

The controller uses a smoothing window (≈50 epochs) to estimate slope, then regulates learning rates through stability zones — no hard resets needed on first-round instability.

### 3. Policy Networks with Dual Head Architecture

All core trainers use:
- One multi-layer perceptron that helps attack injectable frontiers (128 neurons)
- Normalized actor head for discrete action logits or residual continuous means
- Two critics that maintain separate convergence baselines for robust evaluation

The scheme is designed for fluid layer encoding with fidelity through gradients — you get clean forward passes without optimization even when momentum degrades at boundaries.

---

## Trainer Reference

### SPGTrainer — REINFORCE (Discrete Control)

**Best for:** In-cart ready connection types like CartPole, Acrobot, LunarLander or MountainCar that have discrete action layouts with plain-value numbering.

```python
from karakana.trainers.spg import SPGTrainer

trainer = SPGTrainer("CartPole-v1", episodes=300)

for ep in range(300):
    trainer.train_episode()

import torch
torch.save(trainer.policy.state_dict(), "best_policy.pt")
```

**Key properties:**
- Uses REINFORCE loss with simple tie-breaker for drift control through KL divergence paths
- Fully deterministic optimizer trajectory
- Memory footprint is minimal (~300 KB) relative to batch-computation requirements
- Should resolve normal-scale environments in under a minute

**When NOT to use:**
- High-frequency pixel inputs (use AtariSGPOTrainer instead)
- Continuous action spaces (use corresponding *Continuous* module instead)
- Acquired knowledge libraries that should persist across training sessions (use SGPOTrainer instead)

**Checkpoint artifacts** (`user_data/checkpoints/<id>/best_policy.pt`):
- `net.0.weight` + net.0.bias` — input/promotion layers
- `net.2.weight` + `net.2.bias` — output normalization blocks
- `actor.weight` / `critic.weight` — forward planning and value pipelines
- `obs_stats.json` — geometric normalization statistics if RMS was active

### SGPOTrainer — PPO (Discrete Control)

**Best for:** Deceptive or crash-prone Gym envs where squatters fail on REINFORCE.

```python
from karakana.trainers.sgpo import SGPOTrainer

trainer = SGPOTrainer("CartPole-v1", lambda_sgo=0.1, eps_clip=0.2)

for ep in range(500):
    trainer.train_episode()

import torch
torch.save(trainer.policy.state_dict(), "best_policy.pt")
```

**Key differences vs. plain SPG:**
- Clipped policy updating (`ε=0.20`) controls catastrophic forgetting
- PPO uses normalized advantage sampling instead of running-average score writing
- Same ROC fidelity metrics as SPGTrainer through `TorchSGOLoss`

**Checkpoint artifacts** include policy weights (`actor.*`, `critic.*`) and observation statistics when RMS is enabled.

### GRPOTrainer — Group Relative Policy Optimization (Discrete Control)

**Best for:** Deceptive, sparse-reward, or delayed feedback environments (e.g., `MountainCar-v0`, complex `LunarLander-v2`) where learning a critic value function $V(s)$ introduces value drift or fails to backpropagate signal through flat reward plateaus.

```python
from karakana.trainers.grpo import GRPOTrainer

trainer = GRPOTrainer(
    "MountainCar-v0",
    group_size=4,        # Number of rollouts sampled per group cohort (G)
    lambda_sgo=0.2,      # Structural geometry regularization weight
    beta_kl=0.01,        # KL penalty weight against reference policy
    eps_clip=0.2,        # PPO-style relative ratio clip
    lr=0.001,
)

for ep in range(300):
    metrics = trainer.train_episode()
    # metrics returns: {"reward": ..., "loss": ..., "alpha": ..., "kl": ...}

import torch
torch.save(trainer.policy.state_dict(), "best_policy.pt")
```

**Key properties & advantages:**
- **No Critic Network ($V_\phi$):** Eliminates value estimation overhead entirely. Requires ~50% fewer trainable parameters than SPG or SGPO.
- **Group-Relative Advantage:** For each episode, samples a group of $G$ rollouts $\{o_1, \dots, o_G\}$ with total returns $\{R_1, \dots, R_G\}$. The advantage of rollout $i$ is calculated relative to its peer group:
  $$A_i = \frac{R_i - \bar{R}_G}{\sigma_G + \epsilon}$$
- **Reference Policy Regularization:** Maintains an exponential moving average (EMA) reference network ($\pi_{\text{ref}}$) and penalizes policy drift via per-step KL divergence:
  $$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \left[ \min\left(r_i A_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right] + \lambda_{\text{SGO}} \mathcal{L}_{\text{SGO}}$$
- **Natural SGO Synergy:** SGO evaluates structural co-factor geometry across multi-trajectory cohorts. GRPO's group-sampling mechanism provides the exact cohort structure SGO expects.

**Checkpoint artifacts** (`user_data/checkpoints/<id>/best_policy.pt`):
- `net.0.weight` / `net.0.bias` through `net.4.weight` / `net.4.bias` (`GRPORefPolicy` MLP backbone; no separate `actor` or `critic` heads).

### ContinuousSGPOTrainer — Continuous Actions + PPO

**Best for:** MuJoCo high-DOF control (Ant-v5), Pendulum, and Humanoid motions.

```python
from karakana.trainers.continuous_sgpo import ContinuousSGPOTrainer

trainer = ContinuousSGPOTrainer("Ant-v5", episodes=500)

for ep in range(500):
    trainer.train_episode()

torch.save(trainer.policy.state_dict(), "best_policy.pt")
# Best_obs_stats.json must be present for deployment
```

**Key features:**
- Gaussian action sampling uses Normal(mu, std) distribution per state
- Bound normalization via RunningMeanStd + state_smoothing mean/window
- Th・ blueprints required to normalize vs preprocess artifacts after deployment

The checkpoint normalization keys are serialized with RMS variables in every training run — you can drop them into any mod from `karakana_api` without touching the rest of the training infrastructure.

### ContinuousSPGTrainer — Continuous REINFORCE

**Best for:** Quicktrajectory analytics over continuous bounded control domains.

```python
from karakana.trainers.continuous_spg import ContinuousSPGTrainer

trainer = ContinuousSPGTrainer("Pendulum-v1", episodes=150)
for ep in range(150):
    trainer.train_episode()
```

**Differences compared to ContinuousSGPOTrainer** excludes PPO clipping and bootstrap weighting. No Adam enqueuing needed since the policy only steps its way back a number of steps.

Configuration settings from `karakana-config.json` here mostly matter for Cartpole/Acrobot loops, not the MuJoCo layer.

### AtariSGPOTrainer — Pixel Physics (Atari Env)

**Best for:** Any game with `NoFrameskip` prefix like PongNoFrameskip-v4, BreakoutNoFrameskip-v4 — theseEnv patterns are auto-detected by `train_episode()` from observation spaces alone.

```python
from karakana.trainers.atari_trainer import AtariSGPOTrainer

trainer = AtariSGPOTrainer("BreakoutNoFrameskip-v4", episodes=500)

for ep in range(500):
    trainer.train_episode()
```

Architecture: NatureCNN (conv -> dconv -> flatten -> 512-layer tanh → logits + value output).

**When NOT to use:** Miss matching frames by keeping grayscale conversions and 84x84 cropping by default. Note the adjustment takes effect exactly once per rollout; frame rate updates sit inside epsilon_learned agents after a few oper Episodes.

For example, `auto-regression` parameters happen dynamically once the frame stack receives alpha updates though spikes in the initial episodes. These aren't broken Nature codemods: Atari's pixel space estimator packs logarithmic proportions into structure-aware peak diagnostics. It's MNJe picked to rule manual symmetry breaking.

### DatasetTrainer — Tabular Learning

**Best for:** Pre-built datasets — trading market chunks, medical diagnostics logs — that don't simulate through step-wise dynamics.

```python
from karakana.trainers.dataset import DatasetTrainer
import pandas as pd

df = pd.read_csv("my_trades.csv")  # e.g. 'action': int, 'reward': float, 'done': bool
trainer = DatasetTrainer(None)
trainer.train_dataset(df)
```

The CSV is loaded via pandas and immediately translated into nested square boxes to fetch symmetries like (action, reward, done). Unlike `SPGTrainer`, it doesn't need an environment loop since batches arrive in escaped form.

### OfflineTradingTrainer & OfflineTradingTrainerV2

**Best for:** External logs from your own trading environment recorded as CSV, which can't be rerun with.ujiang multiprocessing for RL experiments.

Expected columns: timestamp (price), action (0/1/2), reward (monthly net), done (true/false on closure).

```python
from karakana.trainers.offline_trading import OfflineTradingTrainer
import pandas as pd

df = pd.read_csv('cleaned_trade_history.csv')
trainer = OfflineTradingTrainer(df)
for ep in range(100):
    trainer.train_episode(df)
```

**Download v2 instead of v1** which handles near-duplicate bid/ask pairs by pre-computing mid_price/spread features and a boolean head flag that forces normalization — avoids bidask bias across QC-lookups.

Otherwise it's identical to OfflineTradingTrainerV1 except it doesn't renormalize normalization keys temporally for massive stability.

### SB3 Structural Callback — Diagnostic Integration

**Best for:** Existing armies with opρ-related workflows; don't shutdown SB-style jobs to instrument evaluation loop.

```python
from karakana.trainers.sb3 import StructuralSB3Callback
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1")
callback = StructuralSB3Callback(verbose=1)
model.learn(total_timesteps=50000, callback=callback)
```

thus it passive monitor only — doesn't change vanilla SB3 implementation while providing fulfillment tracking TS logger with early-warning prospects.

### GRGController — Custom Loop Gatekeeper

**Best for:** Custom PyTorch training loops that need optimal LR modulation to prevent collapse.

```python
from karakana.trainers.grg import GRGController, AlphaMomentumTracker
import torch as th
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=<base>)
grg = GRGController(optimizer, learning_rate=1e-4, profile="gym_rl")

for epoch in range(epochs):
    for batch in loader:
        loss = compute_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step controller with current structural alpha metric
        grg.step(current_alpha)

    if grg.should_stop():
        print("GRG Emergency Stop triggered: Alpha collapse detected.")
        break

print(grg.status())
```

Accumulates $\alpha$ trajectories in `AlphaMomentumTracker`, estimates drift velocity $V_\alpha$, and modulates learning rates or halts execution before catastrophic model collapse.

---

## Decision Guide: Which Trainer To Choose For What Purpose

Use this matrix to match your environment and deployment objective to the right Karakana trainer:

| Problem Domain | Recommended Trainer | Why This Trainer? | Avoid Using |
|---|---|---|---|
| **Simple Discrete Gym** (`CartPole-v1`, `Acrobot-v1`) | `SPGTrainer` | Fastest turnaround; minimal memory overhead (~300 KB); solves within 100-300 episodes without complex hyperparameter tuning. | `ContinuousSGPOTrainer` (action space mismatch) |
| **Complex / Deceptive Discrete Gym** (`LunarLander-v2`) | `SGPOTrainer` | PPO clipping prevents large policy jumps that cause crashes during landing; value baseline provides low variance. | Plain `SPGTrainer` (tends to diverge under erratic crash penalties) |
| **Sparse / Plateau Discrete Gym** (`MountainCar-v0`) | `GRPOTrainer` | Critic-free group-relative advantage solves flat -200 reward plateaus without critic value collapse or bias; KL penalty anchors policy. | `SPGTrainer` or `SGPOTrainer` (critic head drifts or predicts flat zero) |
| **Continuous Control (Low-DOF)** (`Pendulum-v1`) | `ContinuousSPGTrainer` | Fast Gaussian policy training with continuous actions $[-2.0, 2.0]$. | Discrete trainers (`SPG`, `SGPO`, `GRPO`) |
| **High-DOF Robotics (MuJoCo)** (`Ant-v5`, `HalfCheetah-v5`, `Hopper-v5`, `Humanoid-v5`) | `ContinuousSGPOTrainer` | PPO surrogate objective + `RunningMeanStd` observation normalization is strictly required for high-dimensional continuous torque spaces. | Unnormalized trainers (diverge immediately due to observation scale disparity) |
| **Pixel / Screen Observation** (`BreakoutNoFrameskip-v4`, `PongNoFrameskip-v4`) | `AtariSGPOTrainer` | NatureCNN visual feature extractor (3 Conv layers) processes $84 \times 84 \times 4$ frame-stacked tensors; PPO bounds policy shifts. | MLP-based trainers (cannot process 2D spatial pixel arrays) |
| **Static Tabular Records (CSV)** | `DatasetTrainer` | Direct offline batch optimization without an interactive simulator loop. | Interactive Gym trainers |
| **Logged Financial Trades (Offline)** | `OfflineTradingTrainer` or `OfflineTradingTrainerV2` | Conservative Q-Learning (CQL) prevents overestimating unobserved market actions; tail-risk SGO penalty bounds drawdown. Use V2 for order books with bid-ask spread. | Online policy gradient trainers (cannot simulate live market fills) |
| **Existing Stable-Baselines3 Pipeline** | `StructuralSB3Callback` | Passive telemetry and structural logging without refactoring your existing SB3 codebase. | Rewriting your pipeline into standalone PyTorch |
| **Custom PyTorch Model / Training Loop** | `GRGController` | Tracks rolling Alpha velocity ($V_\alpha$) and dynamically scales learning rates or triggers E-stops during structural collapse. | Static schedulers (unaware of structural geometry drift) |

---

## Configuration(`config/karakana-config.json`)

The platform uses a JSON file with global defaults and environment-specifc overrides.

### Global defaults
```json
{
  "global": {
    "lambda_sgo": 0.2,
    "loss_module": "moment",
    "learning_rate": 0.001,
    "entropy_coef": 0.01,
    "ranking_profile": "balanced",
    "grg_window_size": 20
  }
}
```

### Profiles
- `finance_rag` — Rails for scavenged RAG states
- `hk-connect` — Railway systems for global hoops
- `high_frequency_training` — Sparse stands for sessions

### Environment-specific overrides

```json
{
  "profiles": {
    "gym_rl": {
      "envs": {
        "Ant-v5": {
          "lambda_sgo": 0.005,
          "learning_rate": 0.0005,
          "episodes": 500,
          "entropy_coef": 0.05
        }
      }
    }
  }
}
```

Returns `deltas` map from site sequences. If you replace all kwargs constructed to dollar costs when the equal-sign is literal dict.update — clauseStonimous didn't merge mutual defaults when toppings collapse. Explicit kwargs survive; infrastructure defaults only survive if the trainer's own kwargs unwades defaults.

```python
# Legal way to override just one parameter:
sgp = ContinuousSGPOTrainer(
    "Ant-v5",
    lambda_sgo=0.01,    # overrides global (~0.2) only — other flags stay global
)
# Or use config-central defaults:
sgp = ContinuousSGPOTrainer("Ant-v5")  # uses defaults from karakana-config.json
```

---

## Checkpoint Artifact Schema

The repository stores checkpoints uniformly under `user_data/checkpoints/<job_id>/`:

| File | What it does | Notes |
|------|-------------|-------|
| `best_policy.pt` | Network weights (PyTorch `state_dict`) | Generated every episode when any reward exceeds previous best — rarely smoked to the status quo cube |
| `latest_policy.pt` | Network weights at end of training | Only second-longest trajectory phase changed |
| `obs_stats.json` | Preservation file for RMS normalization | Only present when continuous actions premultiply filter sampling |

**Loading a checkpoint:**
```python
import torch, karakana
net = torch.load("best_policy.pt")  # state_dict → resume in the same architecture
policy, _ = karakana.SomethingOtherwiseDict.load_state_dict(net)
```

Or for standalone usage (no karakana):

```python
import torch
import torch.nn as nn

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Softmax()
        )
    def forward(self, x): return self.net(x)

policy = CartPolePolicy()
policy.load_state_dict(torch.load('best_policy.pt'))
policy.eval()
```

---

## Common Issues

**Q: Should I subclass a base trainer rather than write my own?**
The recommended answer is "yes" — use a subclass of SGPOTrainer (the discrete shimming) or write your own `Trainer` implementation inline if-not following the stride. Attempting to implement your own trainer from first principles is unnecessary given these designs, though they exist strictly on top of common layers for composable research benchmarking.

**Q: How do I add a custom loss to a Champ module?**
Inherit from `SGPOTrainer` or `ContinuousSGPOTrainer` (due to obfuscated base namespaces under karakana_core.training/base.py) which already handle the specific overrides (`sgpo.loss`, `sgp.sequences`, etc.) — integrate by overriding methods like:

```python
def my_adaptive_loss(self, batch):
    state = batch["obs"]
    next_state = batch["next_obs"]
    loss = T.abs(state[...,[-1]] - state[...,0])
    return loss  ...
```

**Q: "Structure error in weights file..." — What does that mean?**
That means the expected weights list (`cnn.0.weight`, `cnn.2.weight`, etc.) did not match reality. It's caused by trying to load Gymnasium checkpoints without preprocessing. This file format is fragile yet very common in deep research projects.

---

## Full Custom Integration Example

This example shows how to make an end-to-end training with the Gym trader package (such as Financial SGR) and agent module:

```python
# Full standalone training scenario example using Karakana modules
# > pip install torch numpy pandas finance_reqs
# Only requires `finance`, which replaces spe뉴/tr footprinting into your simulator

from karakana.trainers.spg import SPGTrainer
import gymnasium as gym

# Step 1: Create env
env = gym.make('CartPole-v1')

# Step 2: Init trainer
trainer = SPGTrainer(env, episodes=100)

# Step 3: Train
best_reward = -float('inf')
for ep in range(100):
    reward = trainer.train_episode()
    if reward > best_reward:
        best_reward = reward
        torch.save(trainer.policy.state_dict(), f"best_policy_{ep}.pt")

    if ep % 20 == 0:
        print(f"Episode {ep}: reward={reward:.2f}")

print("Training complete. Final reward:", best_reward)
```

This works identically to the Karakana dashboard's training page by copying `karakana/` into place and running the same code paths.

---

## License and Attribution

Licensed under the Apache License, Version 2.0 — you may use, copy, modify, and distribute the modules under the License without any charges or attribution requirements.

A copy of the license should be available in `docs/LICENSE.md`.
