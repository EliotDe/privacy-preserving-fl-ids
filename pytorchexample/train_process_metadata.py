import torch
from dataclasses import dataclass

@dataclass
class TrainProcessMetadata:
    """Contains information needed to evaluate the inversion attack"""
    X_shape: torch.Size
    y_shape: torch.Size
    X: torch.Tensor
    y: torch.Tensor
    train_loss: float
    timestamps: int
