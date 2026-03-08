"""
Train PSMILESNet on Glass Transition Temperature (Tg) Dataset

This script:
1. Loads the polymer SMILES dataset
2. Splits into train/val/test sets
3. Trains the PSMILESNet model
4. Evaluates and generates plots
5. Saves the trained model

Usage:
    python train_tg_psmiles.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from time import time

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psmiles.psmiles_model import PSMILESNet
from psmiles.train_psmiles import PSMILESModel, get_psmiles_model
from psmiles.smiles_featurizer import SMILESCsvLoader
from utils.get_compute_device import get_compute_device

# Set random seeds
RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def load_and_split_data(data_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load the dataset and split into train/val/test sets.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file
    test_size : float
        Fraction for test set
    val_size : float
        Fraction for validation set (from remaining data after test split)
    random_state : int
        Random seed
        
    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        DataFrames for each split
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[''])
    
    # Remove duplicates by averaging
    df_clean = df.groupby('formula').agg({'target': 'mean'}).reset_index()
    print(f"Original samples: {len(df)}, After deduplication: {len(df_clean)}")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df_clean, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir):
    """Save the data splits to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    print(f"Saved splits to {output_dir}/")


def plot_data_distribution(train_df, val_df, test_df, output_dir):
    """Plot the distribution of target values."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [('Train', train_df), ('Validation', val_df), ('Test', test_df)]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, (name, df), color in zip(axes, datasets, colors):
        ax.hist(df['target'], bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Glass Transition Temperature (°C)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Set (n={len(df)})')
        ax.axvline(df['target'].mean(), color='black', linestyle='--', 
                   label=f'Mean: {df["target"].mean():.1f}°C')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tg_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_dir}/tg_distribution.png")


def plot_predictions(actual, predicted, uncertainties, title, output_path):
    """Plot actual vs predicted values with uncertainty."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parity plot
    ax = axes[0]
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    # Color by uncertainty
    scatter = ax.scatter(actual, predicted, c=uncertainties, cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Uncertainty (°C)')
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Tg (°C)')
    ax.set_ylabel('Predicted Tg (°C)')
    ax.set_title(f'{title}\nMAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C, R²: {r2:.3f}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # Residuals plot
    ax = axes[1]
    residuals = predicted - actual
    ax.scatter(actual, residuals, c=uncertainties, cmap='viridis', 
               alpha=0.6, s=30, edgecolors='none')
    ax.axhline(y=0, color='k', linestyle='--', lw=2)
    ax.axhline(y=mae, color='r', linestyle=':', lw=1, label=f'+MAE ({mae:.1f}°C)')
    ax.axhline(y=-mae, color='r', linestyle=':', lw=1, label=f'-MAE')
    
    ax.set_xlabel('Actual Tg (°C)')
    ax.set_ylabel('Residual (Predicted - Actual) (°C)')
    ax.set_title(f'Residuals Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction plot to {output_path}")


def plot_residuals_histogram(actual, predicted, title, output_path):
    """Plot histogram of residuals."""
    residuals = predicted - actual
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.axvline(x=residuals.mean(), color='orange', linestyle='-', lw=2, 
               label=f'Mean: {residuals.mean():.2f}°C')
    
    ax.set_xlabel('Residual (Predicted - Actual) (°C)')
    ax.set_ylabel('Count')
    ax.set_title(f'{title} - Residuals Distribution\nStd: {residuals.std():.2f}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_metrics(results, output_path):
    """Plot summary of metrics across all datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = list(results.keys())
    metrics = ['MAE', 'RMSE', 'R²']
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, metric in enumerate(metrics):
        values = [results[ds][metric] for ds in datasets]
        if metric == 'R²':
            values = [v * 100 for v in values]  # Scale R² to 0-100 for visibility
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i])
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric == 'R²':
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val/100:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Summary')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to {output_path}")


def evaluate_model(model_wrapper, data_path, dataset_name, output_dir):
    """Evaluate the model on a dataset and save results."""
    model_wrapper.load_data(data_path, batch_size=256, train=False)
    actual, predicted, formulas, uncertainties = model_wrapper.predict(model_wrapper.data_loader)
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    print(f"\n{dataset_name} Results:")
    print(f"  MAE:  {mae:.3f} °C")
    print(f"  RMSE: {rmse:.3f} °C")
    print(f"  R²:   {r2:.4f}")
    
    # Save predictions
    df_pred = pd.DataFrame({
        'formula': formulas,
        'actual': actual,
        'predicted': predicted,
        'uncertainty': uncertainties,
        'residual': predicted - actual
    })
    pred_path = f'{output_dir}/predictions_{dataset_name.lower()}.csv'
    df_pred.to_csv(pred_path, index=False)
    print(f"  Saved predictions to {pred_path}")
    
    # Plot predictions
    plot_predictions(
        actual, predicted, uncertainties,
        f'{dataset_name} Set Predictions',
        f'{output_dir}/parity_{dataset_name.lower()}.png'
    )
    
    plot_residuals_histogram(
        actual, predicted,
        f'{dataset_name} Set',
        f'{output_dir}/residuals_{dataset_name.lower()}.png'
    )
    
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2}


def main():
    print("=" * 60)
    print("PSMILESNet Training for Glass Transition Temperature")
    print("=" * 60)
    
    # Configuration
    data_path = 'data/polymer_smiles/glass_transistion_temperature.csv'
    output_dir = 'data/polymer_smiles/tg_splits'
    results_dir = 'psmiles_results'
    model_name = 'psmiles_tg'
    
    # Model hyperparameters
    d_model = 256      # Embedding dimension
    N = 3              # Number of transformer layers
    heads = 4          # Number of attention heads
    n_elements = 10    # Max number of atom types per molecule
    batch_size = 128   # Training batch size
    epochs = 60        # Number of training epochs (should complete in ~20-30 min)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs('figures/lc_data', exist_ok=True)
    
    # Step 1: Load and split data
    print("\n[Step 1] Loading and splitting data...")
    train_df, val_df, test_df = load_and_split_data(data_path)
    save_splits(train_df, val_df, test_df, output_dir)
    
    # Plot data distribution
    plot_data_distribution(train_df, val_df, test_df, results_dir)
    
    # Step 2: Initialize model
    print("\n[Step 2] Initializing model...")
    device = get_compute_device(prefer_last=True)
    print(f"Using device: {device}")
    
    model = get_psmiles_model(
        compute_device=device,
        d_model=d_model,
        N=N,
        heads=heads
    )
    
    wrapper = PSMILESModel(
        model, 
        model_name=model_name,
        n_elements=n_elements,
        verbose=True
    )
    
    # Step 3: Load training data
    print("\n[Step 3] Loading data...")
    train_path = f'{output_dir}/train.csv'
    val_path = f'{output_dir}/val.csv'
    
    wrapper.load_data(train_path, batch_size=batch_size, train=True)
    wrapper.load_data(val_path, batch_size=batch_size, train=False)
    
    # Step 4: Train model
    print("\n[Step 4] Training model...")
    print(f"Training for {epochs} epochs...")
    start_time = time()
    
    best_val_mae = wrapper.fit(epochs=epochs, checkin=2, losscurve=False)
    
    training_time = time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best validation MAE: {best_val_mae:.3f} °C")
    
    # Step 5: Save model
    print("\n[Step 5] Saving model...")
    wrapper.save_network()
    
    # Step 6: Evaluate on all datasets
    print("\n[Step 6] Evaluating model...")
    
    results = {}
    results['Train'] = evaluate_model(wrapper, train_path, 'Train', results_dir)
    results['Validation'] = evaluate_model(wrapper, val_path, 'Validation', results_dir)
    results['Test'] = evaluate_model(wrapper, f'{output_dir}/test.csv', 'Test', results_dir)
    
    # Step 7: Generate summary plots
    print("\n[Step 7] Generating summary plots...")
    plot_summary_metrics(results, f'{results_dir}/metrics_summary.png')
    
    # Save metrics to CSV
    df_metrics = pd.DataFrame(results).T
    df_metrics.to_csv(f'{results_dir}/metrics.csv')
    print(f"Saved metrics to {results_dir}/metrics.csv")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: models/trained_models/{model_name}.pth")
    print(f"Results saved to: {results_dir}/")
    print(f"\nFinal Test Metrics:")
    print(f"  MAE:  {results['Test']['MAE']:.2f} °C")
    print(f"  RMSE: {results['Test']['RMSE']:.2f} °C")
    print(f"  R²:   {results['Test']['R²']:.4f}")
    print("=" * 60)
    
    return wrapper, results


if __name__ == '__main__':
    wrapper, results = main()
