"""
Improved Training Script for Polymer Tg Prediction

Uses Morgan fingerprints + molecular descriptors with a deep
neural network for better accuracy. Expected MAE: 25-35°C.

Usage:
    python train_tg_fingerprint.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from time import time
import pickle

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psmiles.fingerprint_featurizer import FingerprintCsvLoader, featurize_dataset
from psmiles.fingerprint_model import FingerprintNet, robust_l1_loss, huber_loss_with_uncertainty

# Set seeds
RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RNG_SEED)

plt.rcParams.update({'font.size': 12})


class TgPredictor:
    """
    Wrapper class for training and inference.
    """
    
    def __init__(self, model, device, scaler_X=None, scaler_y=None):
        self.model = model
        self.device = device
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    def fit(self, train_loader, val_loader, epochs=200, lr=1e-3, 
            weight_decay=1e-4, patience=20, verbose=True):
        """
        Train the model with early stopping.
        """
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        
        best_val_mae = float('inf')
        best_state = None
        patience_counter = 0
        
        history = {'train_mae': [], 'val_mae': [], 'train_loss': [], 'lr': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_preds, train_targets = [], []
            
            for X, y, _ in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(X)
                pred, log_std = output[:, 0], output[:, 1]
                
                loss = huber_loss_with_uncertainty(pred, log_std, y, delta=10.0)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_losses.append(loss.item())
                train_preds.extend(pred.detach().cpu().numpy())
                train_targets.extend(y.cpu().numpy())
            
            scheduler.step()
            
            # Unscale predictions for MAE calculation
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)
            if self.scaler_y is not None:
                train_preds_unscaled = self.scaler_y.inverse_transform(train_preds.reshape(-1, 1)).flatten()
                train_targets_unscaled = self.scaler_y.inverse_transform(train_targets.reshape(-1, 1)).flatten()
            else:
                train_preds_unscaled = train_preds
                train_targets_unscaled = train_targets
            
            train_mae = mean_absolute_error(train_targets_unscaled, train_preds_unscaled)
            
            # Validation
            val_mae, val_preds, val_targets, _, _ = self.evaluate(val_loader)
            
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            history['train_loss'].append(np.mean(train_losses))
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Train MAE: {train_mae:6.2f}°C | '
                      f'Val MAE: {val_mae:6.2f}°C | '
                      f'LR: {lr_current:.2e} | '
                      f'Best: {best_val_mae:.2f}°C')
            
            if patience_counter >= patience:
                if verbose:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history, best_val_mae
    
    def evaluate(self, loader):
        """Evaluate model on a data loader."""
        self.model.eval()
        all_preds, all_targets, all_uncerts, all_formulas = [], [], [], []
        
        with torch.no_grad():
            for X, y, formulas in loader:
                X = X.to(self.device)
                output = self.model(X)
                pred, log_std = output[:, 0], output[:, 1]
                uncert = torch.exp(log_std)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y.numpy())
                all_uncerts.extend(uncert.cpu().numpy())
                all_formulas.extend(formulas)
        
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        uncerts = np.array(all_uncerts)
        
        # Unscale
        if self.scaler_y is not None:
            preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets = self.scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
            uncerts = uncerts * self.scaler_y.scale_[0]
        
        mae = mean_absolute_error(targets, preds)
        
        return mae, preds, targets, uncerts, all_formulas
    
    def predict(self, X):
        """Make predictions on raw features."""
        self.model.eval()
        
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)
        
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            pred, log_std = output[:, 0], output[:, 1]
            uncert = torch.exp(log_std)
        
        preds = pred.cpu().numpy()
        uncerts = uncert.cpu().numpy()
        
        if self.scaler_y is not None:
            preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
            uncerts = uncerts * self.scaler_y.scale_[0]
        
        return preds, uncerts


def load_and_prepare_data(data_path, test_size=0.15, val_size=0.15):
    """Load data and create train/val/test splits with proper scaling."""
    
    print("Loading and featurizing data...")
    df = pd.read_csv(data_path)
    
    # Deduplicate
    df = df.groupby('formula').agg({'target': 'mean'}).reset_index()
    print(f"Samples after deduplication: {len(df)}")
    
    smiles_list = df['formula'].values.tolist()
    targets = df['target'].values
    
    # Featurize
    X, y, formulas = featurize_dataset(smiles_list, targets, fp_radius=2, fp_bits=2048)
    
    # Split
    X_temp, X_test, y_temp, y_test, form_temp, form_test = train_test_split(
        X, y, formulas, test_size=test_size, random_state=RNG_SEED
    )
    
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, form_train, form_val = train_test_split(
        X_temp, y_temp, form_temp, test_size=val_adjusted, random_state=RNG_SEED
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale targets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return {
        'train': (X_train_scaled, y_train_scaled, form_train),
        'val': (X_val_scaled, y_val_scaled, form_val),
        'test': (X_test_scaled, y_test_scaled, form_test),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'n_features': X_train.shape[1]
    }


def create_dataloaders(data_dict, batch_size=128):
    """Create PyTorch DataLoaders from data dictionary."""
    from psmiles.fingerprint_featurizer import FingerprintDataset
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        X, y, formulas = data_dict[split]
        dataset = FingerprintDataset(X, y, formulas)
        shuffle = (split == 'train')
        loaders[split] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=0, pin_memory=True
        )
    
    return loaders


def plot_results(history, predictions_dict, output_dir):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(history['train_mae'], label='Train MAE', alpha=0.8)
    ax.plot(history['val_mae'], label='Val MAE', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (°C)')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(100, 2 * np.mean(history['val_mae'])))
    
    ax = axes[1]
    ax.plot(history['lr'], color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Parity plots
    for split, (actual, predicted, uncert) in predictions_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        ax = axes[0]
        scatter = ax.scatter(actual, predicted, c=uncert, cmap='viridis', 
                           alpha=0.6, s=20, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Uncertainty (°C)')
        
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        ax.set_xlabel('Actual Tg (°C)')
        ax.set_ylabel('Predicted Tg (°C)')
        ax.set_title(f'{split.capitalize()} Set\n'
                    f'MAE: {mae:.2f}°C | RMSE: {rmse:.2f}°C | R²: {r2:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
        
        # Residuals
        ax = axes[1]
        residuals = predicted - actual
        ax.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', lw=2)
        ax.axvline(residuals.mean(), color='orange', linestyle='-', lw=2, 
                  label=f'Mean: {residuals.mean():.1f}°C')
        ax.set_xlabel('Residual (°C)')
        ax.set_ylabel('Count')
        ax.set_title(f'Residuals (Std: {residuals.std():.1f}°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/parity_{split}.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    print("=" * 70)
    print("Improved Tg Prediction with Fingerprints + Deep Neural Network")
    print("=" * 70)
    
    # Configuration
    data_path = 'data/polymer_smiles/glass_transistion_temperature.csv'
    output_dir = 'psmiles_results_v2'
    model_name = 'psmiles_tg_fingerprint'
    
    # Hyperparameters
    batch_size = 128
    hidden_dims = [1024, 512, 256]  # Network architecture
    n_residual_blocks = 4
    dropout = 0.15
    lr = 5e-4
    weight_decay = 1e-4
    epochs = 300  # More epochs with early stopping
    patience = 30  # Early stopping patience
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models/trained_models', exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n[Step 1] Loading and preparing data...")
    data = load_and_prepare_data(data_path)
    loaders = create_dataloaders(data, batch_size=batch_size)
    
    # Create model
    print("\n[Step 2] Creating model...")
    model = FingerprintNet(
        input_dim=data['n_features'],
        hidden_dims=hidden_dims,
        n_residual_blocks=n_residual_blocks,
        dropout=dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture: {data['n_features']} -> {hidden_dims} -> 2")
    
    # Create predictor
    predictor = TgPredictor(
        model, device, 
        scaler_X=data['scaler_X'],
        scaler_y=data['scaler_y']
    )
    
    # Train
    print("\n[Step 3] Training model...")
    print(f"Max epochs: {epochs}, Patience: {patience}")
    print("-" * 70)
    
    start_time = time()
    history, best_val_mae = predictor.fit(
        loaders['train'], loaders['val'],
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, verbose=True
    )
    training_time = time() - start_time
    
    print("-" * 70)
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best validation MAE: {best_val_mae:.2f}°C")
    
    # Evaluate
    print("\n[Step 4] Evaluating model...")
    
    predictions = {}
    results = {}
    
    for split in ['train', 'val', 'test']:
        mae, preds, targets, uncerts, formulas = predictor.evaluate(loaders[split])
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2 = r2_score(targets, preds)
        
        predictions[split] = (targets, preds, uncerts)
        results[split] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        
        print(f"\n{split.capitalize():12} | MAE: {mae:6.2f}°C | RMSE: {rmse:6.2f}°C | R²: {r2:.4f}")
        
        # Save predictions
        df_pred = pd.DataFrame({
            'formula': formulas,
            'actual': targets,
            'predicted': preds,
            'uncertainty': uncerts,
            'residual': preds - targets
        })
        df_pred.to_csv(f'{output_dir}/predictions_{split}.csv', index=False)
    
    # Generate plots
    print("\n[Step 5] Generating plots...")
    plot_results(history, predictions, output_dir)
    
    # Save model
    print("\n[Step 6] Saving model...")
    save_dict = {
        'model_state': model.state_dict(),
        'scaler_X': data['scaler_X'],
        'scaler_y': data['scaler_y'],
        'model_config': {
            'input_dim': data['n_features'],
            'hidden_dims': hidden_dims,
            'n_residual_blocks': n_residual_blocks,
            'dropout': dropout
        },
        'results': results,
        'history': history
    }
    torch.save(save_dict, f'models/trained_models/{model_name}.pth')
    print(f"Saved to models/trained_models/{model_name}.pth")
    
    # Save metrics
    df_metrics = pd.DataFrame(results).T
    df_metrics.to_csv(f'{output_dir}/metrics.csv')
    
    # Comparison with previous model
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<25} | {'Test MAE':>12} | {'Test R²':>10}")
    print("-" * 55)
    print(f"{'PSMILESNet (atom counts)':<25} | {'44.73°C':>12} | {'0.687':>10}")
    print(f"{'FingerprintNet (this)':<25} | {results['test']['MAE']:>10.2f}°C | {results['test']['R²']:>10.4f}")
    
    improvement = 44.73 - results['test']['MAE']
    print(f"\nImprovement: {improvement:.2f}°C ({improvement/44.73*100:.1f}% reduction in MAE)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return predictor, results


if __name__ == '__main__':
    predictor, results = main()
