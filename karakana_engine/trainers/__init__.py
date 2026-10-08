"""
Karakana Reinforcement Learning Trainers
Differentiable SGO-coupled policy optimization algorithms.
"""

from .atari_trainer import AtariSGPOTrainer
from .continuous_sgpo import ContinuousSGPOTrainer
from .continuous_spg import ContinuousSPGTrainer
from .dataset import DatasetTrainer
from .grg import AlphaMomentumTracker, GRGController
from .grpo import GRPORefPolicy, GRPOTrainer
from .sb3 import StructuralSB3Callback
from .sgpo import SGPOTrainer
from .spg import SPGTrainer

__all__ = [
    "SPGTrainer",
    "SGPOTrainer",
    "GRPOTrainer",
    "GRPORefPolicy",
    "ContinuousSPGTrainer",
    "ContinuousSGPOTrainer",
    "AtariSGPOTrainer",
    "DatasetTrainer",
    "StructuralSB3Callback",
    "GRGController",
    "AlphaMomentumTracker",
]
