"""
Improved Neural Network for Polymer Tg Prediction

Uses Morgan fingerprints and molecular descriptors with a deep
residual neural network for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = x + residual
        x = self.act(x)
        return x


class FingerprintNet(nn.Module):
    """
    Deep neural network for fingerprint-based property prediction.
    
    Architecture:
    - Input projection with batch norm
    - Multiple residual blocks
    - Output heads for prediction and uncertainty
    """
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], 
                 n_residual_blocks=3, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Input projection
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.input_layers = nn.Sequential(*layers)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], dropout=dropout)
            for _ in range(n_residual_blocks)
        ])
        
        # Output head - predicts mean and log_std
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 2)  # [mean, log_std]
        )
    
    def forward(self, x):
        x = self.input_layers(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        output = self.output_head(x)
        return output


class EnsembleFingerprintNet(nn.Module):
    """
    Ensemble of FingerprintNet models for improved predictions.
    """
    
    def __init__(self, input_dim, n_models=3, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            FingerprintNet(input_dim, **kwargs)
            for _ in range(n_models)
        ])
        self.n_models = n_models
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0)  # Average predictions


def robust_l1_loss(prediction, log_std, target):
    """
    Robust L1 loss with learned uncertainty.
    """
    absolute = torch.abs(prediction - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def mse_loss_with_uncertainty(prediction, log_std, target):
    """
    MSE loss with learned uncertainty (Gaussian NLL).
    """
    squared = (prediction - target) ** 2
    loss = 0.5 * squared * torch.exp(-2 * log_std) + log_std
    return torch.mean(loss)


def huber_loss_with_uncertainty(prediction, log_std, target, delta=1.0):
    """
    Huber loss with learned uncertainty - robust to outliers.
    """
    diff = prediction - target
    abs_diff = torch.abs(diff)
    
    quadratic = 0.5 * diff ** 2
    linear = delta * (abs_diff - 0.5 * delta)
    
    loss = torch.where(abs_diff < delta, quadratic, linear)
    loss = loss * torch.exp(-log_std) + 0.5 * log_std
    return torch.mean(loss)


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    input_dim = 2048 + 25  # fingerprint + descriptors
    model = FingerprintNet(input_dim).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(32, input_dim).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
