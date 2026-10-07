# Karakana Engine

High-performance PyTorch Reinforcement Learning framework with **Structural Geometry Optimization (SGO)**.

Karakana trainers couple policy gradients with differentiable geometric spectral bounds to prevent catastrophic collapse and policy fragility.

---

## ⚡ Quickstart

```bash
pip install karakana-engine
# With environment extras:
pip install "karakana-engine[box2d,mujoco,atari]"
```

### Basic Training (SPG / REINFORCE)

```python
from karakana.trainers import SPGTrainer

trainer = SPGTrainer("CartPole-v1", episodes=300)

for ep in range(300):
    reward, loss = trainer.train_episode()
    if ep % 50 == 0:
        print(f"Episode {ep}: reward={reward:.1f}, alpha={trainer.current_alpha:.3f}")
```

### PPO with SGO Bounds (SGPO)

```python
from karakana.trainers import SGPOTrainer

trainer = SGPOTrainer("LunarLander-v2", lambda_sgo=0.1, eps_clip=0.2)

for ep in range(500):
    reward, loss = trainer.train_episode()
```

### Critic-Free Group Relative Policy Optimization (GRPO)

```python
from karakana.trainers import GRPOTrainer

# Ideal for deceptive landscapes or sparse rewards (e.g. MountainCar)
trainer = GRPOTrainer("MountainCar-v0", group_size=4, lambda_sgo=0.2, beta_kl=0.01)

for ep in range(300):
    reward, loss = trainer.train_episode()
```

### MuJoCo High-DOF Continuous Robotics

```python
from karakana.trainers import ContinuousSGPOTrainer

trainer = ContinuousSGPOTrainer("Ant-v5", episodes=500)

for ep in range(500):
    reward, loss = trainer.train_episode()
```

---

## 🧠 Available Trainers

| Trainer | Class | Primary Domain | Algorithm |
|---|---|---|---|
| **SPG** | `SPGTrainer` | Classic Gym (CartPole, Acrobot) | Policy Gradient + SGO spectral penalty |
| **SGPO** | `SGPOTrainer` | Crash-prone Gym (LunarLander) | PPO ratio clipping + SGO bounds |
| **GRPO** | `GRPOTrainer` | Sparse / Deceptive (MountainCar) | Critic-free group advantage + KL reference penalty |
| **Continuous SPG** | `ContinuousSPGTrainer` | Bounded Continuous (Pendulum) | Gaussian head + REINFORCE |
| **Continuous SGPO** | `ContinuousSGPOTrainer` | Robotics (Ant, HalfCheetah, Humanoid) | Continuous PPO + RunningMeanStd normalization |
| **Atari SGPO** | `AtariSGPOTrainer` | Pixel Games (Breakout, Pong) | NatureCNN (3 Conv layers) + PPO |
| **Dataset** | `DatasetTrainer` | Offline Tabular CSVs | Offline gradient steps |
| **SB3 Callback** | `StructuralSB3Callback` | Stable-Baselines3 integration | Real-time passive SGO telemetry |
| **GRG Controller** | `GRGController` | Custom PyTorch loops | Alpha-Drift Velocity ($V_\alpha$) governor & E-Stop |

Detailed guide & decision matrix: see [`docs/TRAINERS.md`](docs/TRAINERS.md).
Theoretical mathematical specifications: see [`docs/THEORY.md`](docs/THEORY.md).

---

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).
