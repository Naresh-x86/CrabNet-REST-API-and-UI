"""
PSMILESNet: CrabNet adapted for Polymer SMILES prediction.

This module contains the neural network architecture for predicting
properties from polymer SMILES strings.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from psmiles.smiles_featurizer import (
    ATOM_SYMBOLS, ATOM_FEATURES, NUM_ATOM_TYPES, 
    get_atom_feature_matrix
)

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network (from Roost/CrabNet).
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)


class SMILESEmbedder(nn.Module):
    """
    Embedder for SMILES atom types.
    Uses learned embeddings initialized with chemical properties.
    """
    
    def __init__(self, d_model, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device
        
        # Get atom features
        atom_features = get_atom_feature_matrix()
        feat_size = atom_features.shape[-1]
        
        # Create embedding from atom features
        self.fc_embed = nn.Linear(feat_size, d_model).to(self.compute_device)
        
        # Create pretrained embedding from atom features
        atom_features_tensor = torch.as_tensor(atom_features, dtype=data_type_torch)
        self.atom_features = nn.Embedding.from_pretrained(
            atom_features_tensor, freeze=False
        ).to(self.compute_device, dtype=data_type_torch)
    
    def forward(self, src):
        """
        src: (batch_size, n_elements) - atom type indices
        """
        atom_emb = self.atom_features(src)  # (batch, n_elem, feat_size)
        x_emb = self.fc_embed(atom_emb)     # (batch, n_elem, d_model)
        return x_emb


class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using positional encoding.
    (Same as CrabNet)
    """
    
    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1, self.resolution,
                           requires_grad=False).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model,
                                  requires_grad=False).view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x.clamp(min=1e-8)))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        frac_idx = torch.clamp(frac_idx, min=0, max=self.resolution-1)
        out = self.pe[frac_idx]
        return out


class SMILESEncoder(nn.Module):
    """
    Transformer encoder for SMILES atom sequences.
    """
    
    def __init__(self, d_model, N, heads, frac=False, attn=True, compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.compute_device = compute_device
        
        self.embed = SMILESEmbedder(d_model=self.d_model, compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)
        
        self.emb_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.]))
        
        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model, nhead=self.heads,
                dim_feedforward=2048, dropout=0.1
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N)

    def forward(self, src, frac):
        """
        src: (batch_size, n_elements) - atom type indices
        frac: (batch_size, n_elements) - atom fractions
        """
        x = self.embed(src) * 2**self.emb_scaler
        
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1
        
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler
        
        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)
        
        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)
        
        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)
        
        return x


class PSMILESNet(nn.Module):
    """
    PSMILESNet: CrabNet adapted for Polymer SMILES prediction.
    
    Architecture:
    - SMILES atom embeddings
    - Transformer encoder with attention
    - Residual network for output
    """
    
    def __init__(self, out_dims=3, d_model=256, N=3, heads=4, 
                 compute_device=None, residual_nn='roost'):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        
        self.encoder = SMILESEncoder(
            d_model=self.d_model, N=self.N, heads=self.heads,
            compute_device=self.compute_device
        )
        
        if residual_nn == 'roost':
            self.out_hidden = [512, 256, 128, 64]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
        else:
            self.out_hidden = [128, 64]
            self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)

    def forward(self, src, frac):
        output = self.encoder(src, frac)
        
        # Average the atom contributions
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)
        
        if self.avg:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        
        return output


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PSMILESNet(out_dims=3, d_model=256, N=3, heads=4, compute_device=device)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    n_elements = 8
    src = torch.randint(0, NUM_ATOM_TYPES, (batch_size, n_elements)).to(device)
    frac = torch.rand(batch_size, n_elements).to(device)
    
    output = model(src, frac)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params:,}")
