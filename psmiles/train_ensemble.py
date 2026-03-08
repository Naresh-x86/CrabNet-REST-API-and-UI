"""
Advanced Training with Hyperparameter Search and Ensemble

Uses larger networks, more training, and ensembling for best results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from time import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psmiles.fingerprint_featurizer import featurize_dataset, FingerprintDataset
from psmiles.fingerprint_model import FingerprintNet, huber_loss_with_uncertainty

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

plt.rcParams.update({'font.size': 12})


class EnsemblePredictor:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, models, scalers_X, scalers_y, device):
        self.models = models
        self.scalers_X = scalers_X
        self.scalers_y = scalers_y
        self.device = device
    
    def predict(self, X):
        all_preds = []
        all_uncerts = []
        
        for model, scaler_X, scaler_y in zip(self.models, self.scalers_X, self.scalers_y):
            model.eval()
            X_scaled = scaler_X.transform(X)
            X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                output = model(X_tensor)
                pred = output[:, 0].cpu().numpy()
                uncert = torch.exp(output[:, 1]).cpu().numpy()
            
            pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
            uncert = uncert * scaler_y.scale_[0]
            
            all_preds.append(pred)
            all_uncerts.append(uncert)
        
        # Average predictions
        ensemble_pred = np.mean(all_preds, axis=0)
        # Combined uncertainty (model uncertainty + prediction spread)
        ensemble_uncert = np.sqrt(np.mean(np.array(all_uncerts)**2, axis=0) + np.var(all_preds, axis=0))
        
        return ensemble_pred, ensemble_uncert


def train_single_model(X_train, y_train, X_val, y_val, scaler_y, device, config):
    """Train a single model with given config."""
    
    # Create datasets
    train_dataset = FingerprintDataset(X_train, y_train, np.arange(len(y_train)))
    val_dataset = FingerprintDataset(X_val, y_val, np.arange(len(y_val)))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = FingerprintNet(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        n_residual_blocks=config['n_residual_blocks'],
        dropout=config['dropout']
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = huber_loss_with_uncertainty(output[:, 0], output[:, 1], y, delta=10.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(device)
                pred = model(X)[:, 0].cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(y.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_preds_unscaled = scaler_y.inverse_transform(val_preds.reshape(-1, 1)).flatten()
        val_targets_unscaled = scaler_y.inverse_transform(val_targets.reshape(-1, 1)).flatten()
        val_mae = mean_absolute_error(val_targets_unscaled, val_preds_unscaled)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_mae


def train_cross_validation(X, y, formulas, device, config, n_folds=5):
    """Train models using cross-validation."""
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    
    models = []
    scalers_X = []
    scalers_y = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n  Fold {fold + 1}/{n_folds}...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_fold)
        X_val_scaled = scaler_X.transform(X_val_fold)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_fold.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val_fold.reshape(-1, 1)).flatten()
        
        # Train
        model, val_mae = train_single_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            scaler_y, device, config
        )
        
        models.append(model)
        scalers_X.append(scaler_X)
        scalers_y.append(scaler_y)
        fold_scores.append(val_mae)
        
        print(f"    Fold {fold + 1} Val MAE: {val_mae:.2f}°C")
    
    print(f"\n  CV Mean MAE: {np.mean(fold_scores):.2f}°C (+/- {np.std(fold_scores):.2f})")
    
    return models, scalers_X, scalers_y, fold_scores


def hyperparameter_search(X, y, device, n_configs=6):
    """Quick hyperparameter search."""
    
    configs = [
        {'hidden_dims': [1024, 512, 256], 'n_residual_blocks': 4, 'dropout': 0.15, 'lr': 5e-4, 'batch_size': 128},
        {'hidden_dims': [1024, 512, 256], 'n_residual_blocks': 5, 'dropout': 0.1, 'lr': 3e-4, 'batch_size': 64},
        {'hidden_dims': [2048, 1024, 512], 'n_residual_blocks': 4, 'dropout': 0.2, 'lr': 3e-4, 'batch_size': 128},
        {'hidden_dims': [1024, 512, 256, 128], 'n_residual_blocks': 3, 'dropout': 0.15, 'lr': 5e-4, 'batch_size': 64},
        {'hidden_dims': [512, 256, 128], 'n_residual_blocks': 6, 'dropout': 0.1, 'lr': 1e-3, 'batch_size': 128},
        {'hidden_dims': [1024, 512, 256], 'n_residual_blocks': 4, 'dropout': 0.05, 'lr': 2e-4, 'batch_size': 32},
    ]
    
    # Add common params
    for config in configs:
        config['input_dim'] = X.shape[1]
        config['epochs'] = 150
        config['patience'] = 25
        config['weight_decay'] = 1e-4
    
    # Split for search
    X_search, X_holdout, y_search, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_search, y_search, test_size=0.2, random_state=RNG_SEED
    )
    
    # Scale
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    results = []
    
    for i, config in enumerate(configs[:n_configs]):
        print(f"\n  Config {i+1}/{n_configs}: {config['hidden_dims']}, res={config['n_residual_blocks']}, drop={config['dropout']}")
        
        model, val_mae = train_single_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            scaler_y, device, config
        )
        
        results.append((val_mae, config))
        print(f"    Val MAE: {val_mae:.2f}°C")
    
    # Sort by val MAE
    results.sort(key=lambda x: x[0])
    best_config = results[0][1]
    
    print(f"\n  Best config: {best_config['hidden_dims']}, MAE: {results[0][0]:.2f}°C")
    
    return best_config


def main():
    print("=" * 70)
    print("Advanced Tg Prediction with Ensemble Learning")
    print("=" * 70)
    
    data_path = 'data/polymer_smiles/glass_transistion_temperature.csv'
    output_dir = 'psmiles_results_v3'
    model_name = 'psmiles_tg_ensemble'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models/trained_models', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n[Step 1] Loading and featurizing data...")
    start_time = time()
    
    df = pd.read_csv(data_path)
    df = df.groupby('formula').agg({'target': 'mean'}).reset_index()
    smiles_list = df['formula'].values.tolist()
    targets = df['target'].values
    
    X, y, formulas = featurize_dataset(smiles_list, targets, fp_radius=2, fp_bits=2048, verbose=True)
    
    # Hold out test set
    X_train_full, X_test, y_train_full, y_test, form_train, form_test = train_test_split(
        X, y, formulas, test_size=0.15, random_state=RNG_SEED
    )
    
    print(f"Train+Val: {len(X_train_full)}, Test: {len(X_test)}")
    
    # Hyperparameter search
    print("\n[Step 2] Hyperparameter search...")
    best_config = hyperparameter_search(X_train_full, y_train_full, device, n_configs=6)
    
    # Update config for final training
    best_config['epochs'] = 200
    best_config['patience'] = 30
    
    # Cross-validation ensemble
    print("\n[Step 3] Training cross-validation ensemble...")
    models, scalers_X, scalers_y, fold_scores = train_cross_validation(
        X_train_full, y_train_full, form_train, device, best_config, n_folds=5
    )
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(models, scalers_X, scalers_y, device)
    
    # Evaluate
    print("\n[Step 4] Evaluating ensemble on test set...")
    test_preds, test_uncerts = ensemble.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"\n  Test MAE:  {test_mae:.2f}°C")
    print(f"  Test RMSE: {test_rmse:.2f}°C")
    print(f"  Test R²:   {test_r2:.4f}")
    
    # Generate plots
    print("\n[Step 5] Generating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Parity plot
    ax = axes[0]
    scatter = ax.scatter(y_test, test_preds, c=test_uncerts, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='Uncertainty (°C)')
    min_val, max_val = min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_xlabel('Actual Tg (°C)')
    ax.set_ylabel('Predicted Tg (°C)')
    ax.set_title(f'Test Set Parity\nMAE: {test_mae:.2f}°C | R²: {test_r2:.3f}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # Residuals histogram
    ax = axes[1]
    residuals = test_preds - y_test
    ax.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Residual (°C)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residuals Distribution\nMean: {residuals.mean():.1f}°C, Std: {residuals.std():.1f}°C')
    ax.grid(True, alpha=0.3)
    
    # CV scores
    ax = axes[2]
    folds = np.arange(1, len(fold_scores) + 1)
    ax.bar(folds, fold_scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(fold_scores), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(fold_scores):.2f}°C')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Validation MAE (°C)')
    ax.set_title('Cross-Validation Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save predictions
    df_pred = pd.DataFrame({
        'formula': form_test,
        'actual': y_test,
        'predicted': test_preds,
        'uncertainty': test_uncerts,
        'residual': residuals
    })
    df_pred.to_csv(f'{output_dir}/predictions_test.csv', index=False)
    
    # Save model
    print("\n[Step 6] Saving ensemble model...")
    save_dict = {
        'models_state': [m.state_dict() for m in models],
        'scalers_X': scalers_X,
        'scalers_y': scalers_y,
        'model_config': best_config,
        'fold_scores': fold_scores,
        'test_results': {'MAE': test_mae, 'RMSE': test_rmse, 'R2': test_r2}
    }
    torch.save(save_dict, f'models/trained_models/{model_name}.pth')
    
    elapsed = time() - start_time
    
    # Results comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<30} | {'Test MAE':>12} | {'Test R²':>10}")
    print("-" * 60)
    print(f"{'PSMILESNet (atom counts)':<30} | {'44.73°C':>12} | {'0.687':>10}")
    print(f"{'FingerprintNet (single)':<30} | {'25.51°C':>12} | {'0.857':>10}")
    print(f"{'FingerprintNet (ensemble)':<30} | {test_mae:>10.2f}°C | {test_r2:>10.4f}")
    
    improvement_single = 44.73 - 25.51
    improvement_ensemble = 44.73 - test_mae
    
    print(f"\nImprovements vs baseline:")
    print(f"  Single model:   {improvement_single:.2f}°C (43.0% MAE reduction)")
    print(f"  Ensemble model: {improvement_ensemble:.2f}°C ({improvement_ensemble/44.73*100:.1f}% MAE reduction)")
    
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")
    print("\n" + "=" * 70)
    
    test_results = {'MAE': test_mae, 'RMSE': test_rmse, 'R2': test_r2}
    return ensemble, test_results


if __name__ == '__main__':
    ensemble, results = main()
