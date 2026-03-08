"""
Training wrapper for PSMILESNet model.

This module provides the Model class that handles:
- Data loading
- Training loop
- Prediction
- Model saving/loading
"""

import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from torch.optim.lr_scheduler import CyclicLR

from psmiles.psmiles_model import PSMILESNet
from psmiles.smiles_featurizer import SMILESCsvLoader

# Import utilities from CrabNet
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import Lamb, Lookahead, RobustL1, Scaler, DummyScaler, count_parameters
from utils.get_compute_device import get_compute_device
from utils.optim import SWA

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


class PSMILESModel:
    """
    Model wrapper for PSMILESNet training and inference.
    """
    
    def __init__(self, model, model_name='PSMILESNet', n_elements=8, 
                 verbose=True, scale=True):
        self.model = model
        self.model_name = model_name
        self.n_elements = n_elements
        self.compute_device = model.compute_device
        self.fudge = 0.02  # Fractional tolerance for jitter
        self.verbose = verbose
        self.scale = scale
        self.classification = False
        
        self.train_loader = None
        self.data_loader = None
        self.scaler = None
        
        if self.compute_device is None:
            self.compute_device = get_compute_device()
        
        if self.verbose:
            print('\nPSMILESNet architecture: out_dims, d_model, N, heads')
            print(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
            print(f'Running on device: {self.compute_device}')
            print(f'Model size: {count_parameters(self.model):,} parameters\n')
    
    def load_data(self, file_name, batch_size=128, train=False):
        """Load data from CSV file."""
        self.batch_size = batch_size
        inference = not train
        
        data_loaders = SMILESCsvLoader(
            csv_data=file_name,
            batch_size=batch_size,
            n_elements=self.n_elements,
            inference=inference,
            verbose=self.verbose,
            scale=self.scale
        )
        
        if self.verbose:
            print(f'Loaded data with {data_loaders.n_train} samples, '
                  f'{data_loaders.n_elements} max atom types')
        
        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.data[1]
        
        if train:
            self.train_len = len(y)
            self.scaler = Scaler(y)
            self.train_loader = data_loader
        
        self.data_loader = data_loader
    
    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        ti = time()
        minima = []
        
        for i, data in enumerate(self.train_loader):
            X, y, formula = data
            y = self.scaler.scale(y)
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            
            # Add jitter to fractions
            frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac_sum = frac.sum(dim=1).unsqueeze(1)
            frac_sum[frac_sum == 0] = 1  # Avoid division by zero
            frac = frac / frac_sum.repeat(1, frac.shape[-1])
            
            src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
            y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
            
            output = self.model.forward(src, frac)
            prediction, uncertainty = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), uncertainty.view(-1), y.view(-1))
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.stepping:
                self.lr_scheduler.step()
            
            # SWA update check
            swa_check = (self.epochs_step * self.swa_start - 1)
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            
            if learning_time:
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.optimizer.update_swa(mae_v)
                minima.append(self.optimizer.minimum_found)
        
        if learning_time and not any(minima):
            self.optimizer.discard_count += 1
            if self.verbose:
                print(f'Epoch {self.epoch} failed to improve.')
                print(f'Discarded: {self.optimizer.discard_count}/{self.discard_n} weight updates')
        
        return time() - ti
    
    def fit(self, epochs=40, checkin=None, losscurve=False):
        """Train the model."""
        assert self.train_loader is not None, 'Load training data first'
        assert self.data_loader is not None, 'Load validation data first'
        
        self.loss_curve = {'train': [], 'val': []}
        self.epochs_step = 1
        self.step_size = self.epochs_step * len(self.train_loader)
        
        if self.verbose:
            print(f'Stepping every {self.step_size} training passes, '
                  f'cycling lr every {self.epochs_step} epochs')
        
        if checkin is None:
            checkin = self.epochs_step * 2
        
        self.step_count = 0
        self.criterion = RobustL1
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        
        lr_scheduler = CyclicLR(
            self.optimizer, base_lr=1e-4, max_lr=6e-3,
            cycle_momentum=False, step_size_up=self.step_size
        )
        
        self.swa_start = 2
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = 3
        
        best_val_mae = float('inf')
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs
            
            epoch_time = self.train_epoch()
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])
            
            if (epoch + 1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                
                if self.verbose:
                    print(f'Epoch: {epoch}/{epochs} --- train mae: {mae_t:.3f}, val mae: {mae_v:.3f}')
                
                if mae_v < best_val_mae:
                    best_val_mae = mae_v
                
                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch + 1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)
                
                if losscurve:
                    self._plot_loss_curve(checkin)
            
            if self.optimizer.discard_count >= self.discard_n:
                if self.verbose:
                    print(f'Early stopping: {self.optimizer.discard_count}/{self.discard_n} '
                          f'weight updates discarded')
                self.optimizer.swap_swa_sgd()
                break
        
        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()
        
        # Save final loss curve
        self._save_loss_curve(checkin)
        
        return best_val_mae
    
    def _plot_loss_curve(self, checkin):
        """Plot current loss curve."""
        plt.figure(figsize=(10, 6))
        xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
        xval[0] = 0
        plt.plot(xval, self.loss_curve['train'], 'o-', label='train MAE', markersize=4)
        plt.plot(xval, self.loss_curve['val'], 's--', label='val MAE', markersize=4)
        if self.xswa:
            plt.plot(self.xswa, self.yswa, 'o', ms=12, mfc='none', label='SWA point')
        plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
        plt.title(f'{self.model_name} - Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _save_loss_curve(self, checkin):
        """Save loss curve data and plot."""
        os.makedirs('figures/lc_data', exist_ok=True)
        
        xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
        xval[0] = 0
        
        df_loss = pd.DataFrame({
            'epoch': xval,
            'train_loss': self.loss_curve['train'],
            'val_loss': self.loss_curve['val'],
            'swa': ['y' if e in self.xswa else 'n' for e in xval]
        })
        df_loss.to_csv(f'figures/lc_data/{self.model_name}_lc.csv', index=False)
        
        # Save plot
        plt.figure(figsize=(10, 6))
        plt.plot(xval, self.loss_curve['train'], 'o-', label='train MAE', markersize=4)
        plt.plot(xval, self.loss_curve['val'], 's--', label='val MAE', markersize=4)
        if self.xswa:
            plt.plot(self.xswa, self.yswa, 'o', ms=12, mfc='none', label='SWA point')
        plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
        plt.title(f'{self.model_name} - Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('MAE (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'figures/lc_data/{self.model_name}_lc.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def predict(self, loader):
        """Make predictions on a data loader."""
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0]) / 2)
        
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=object)
        
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                
                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                
                data_loc = slice(i * self.batch_size, i * self.batch_size + len(y), 1)
                
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy().astype('float32')
                formulae[data_loc] = formula
        
        self.model.train()
        
        return (act, pred, formulae, uncert)
    
    def save_network(self, path=None):
        """Save the model to disk."""
        os.makedirs('models/trained_models', exist_ok=True)
        
        if path is None:
            path = f'models/trained_models/{self.model_name}.pth'
        
        save_dict = {
            'weights': self.model.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'model_name': self.model_name,
            'n_elements': self.n_elements,
            'model_config': {
                'out_dims': self.model.out_dims,
                'd_model': self.model.d_model,
                'N': self.model.N,
                'heads': self.model.heads,
            }
        }
        torch.save(save_dict, path)
        print(f'Model saved to {path}')
    
    def load_network(self, path):
        """Load a saved model from disk."""
        if not path.startswith('models/'):
            path = f'models/trained_models/{path}'
        
        network = torch.load(path, map_location=self.compute_device)
        
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(network['weights'])
        self.scaler.load_state_dict(network['scaler_state'])
        self.model_name = network['model_name']
        
        if 'n_elements' in network:
            self.n_elements = network['n_elements']
        
        print(f'Model loaded from {path}')


def get_psmiles_model(compute_device=None, d_model=256, N=3, heads=4):
    """Create a new PSMILESNet model."""
    if compute_device is None:
        compute_device = get_compute_device()
    
    model = PSMILESNet(
        out_dims=3,
        d_model=d_model,
        N=N,
        heads=heads,
        compute_device=compute_device,
        residual_nn='roost'
    ).to(compute_device)
    
    return model


if __name__ == '__main__':
    # Quick test
    device = get_compute_device()
    print(f"Using device: {device}")
    
    model = get_psmiles_model(compute_device=device)
    wrapper = PSMILESModel(model, model_name='test_psmiles', verbose=True)
