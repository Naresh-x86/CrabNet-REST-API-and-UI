"""
PSMILESNet Inference Script

Use this script to make predictions on new polymer SMILES data
using the trained PSMILESNet model.

Usage:
    python predict_tg.py --input new_polymers.csv --output predictions.csv
    
Or in Python:
    from psmiles.predict_tg import predict_tg
    predictions = predict_tg(['*CC(*)c1ccccc1', '*CC(*)OC'])
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psmiles.psmiles_model import PSMILESNet
from psmiles.train_psmiles import PSMILESModel
from psmiles.smiles_featurizer import smiles_to_edm, SMILESCsvLoader
from utils.get_compute_device import get_compute_device
from utils.utils import Scaler


def load_trained_model(model_path='models/trained_models/psmiles_tg.pth', device=None):
    """
    Load the trained PSMILESNet model.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model (.pth file)
    device : torch.device, optional
        Device to load model on (defaults to best available)
        
    Returns
    -------
    wrapper : PSMILESModel
        The model wrapper ready for inference
    """
    if device is None:
        device = get_compute_device()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config
    config = checkpoint.get('model_config', {})
    n_elements = checkpoint.get('n_elements', 10)
    
    # Create model
    model = PSMILESNet(
        out_dims=config.get('out_dims', 3),
        d_model=config.get('d_model', 256),
        N=config.get('N', 3),
        heads=config.get('heads', 4),
        compute_device=device
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['weights'])
    
    # Create wrapper
    wrapper = PSMILESModel(
        model,
        model_name=checkpoint.get('model_name', 'psmiles_tg'),
        n_elements=n_elements,
        verbose=False
    )
    
    # Load scaler
    wrapper.scaler = Scaler(torch.zeros(3))
    wrapper.scaler.load_state_dict(checkpoint['scaler_state'])
    
    print(f"Loaded model from {model_path}")
    print(f"Model: {checkpoint.get('model_name', 'psmiles_tg')}")
    
    return wrapper


def predict_tg(smiles_list, model_wrapper=None, return_uncertainty=True):
    """
    Predict glass transition temperature for a list of polymer SMILES.
    
    Parameters
    ----------
    smiles_list : list of str
        List of polymer SMILES strings
    model_wrapper : PSMILESModel, optional
        Pre-loaded model wrapper. If None, will load default model.
    return_uncertainty : bool
        Whether to return uncertainty estimates
        
    Returns
    -------
    predictions : np.ndarray
        Predicted Tg values (°C)
    uncertainties : np.ndarray, optional
        Uncertainty estimates (°C), if return_uncertainty=True
    """
    if model_wrapper is None:
        model_wrapper = load_trained_model()
    
    device = model_wrapper.compute_device
    n_elements = model_wrapper.n_elements
    
    # Create dummy targets (not used for inference)
    dummy_targets = np.zeros(len(smiles_list))
    
    # Convert SMILES to EDM format
    X, _, formulas = smiles_to_edm(
        smiles_list, dummy_targets,
        n_elements=n_elements, verbose=False, scale=True
    )
    
    # Prepare data
    X = torch.as_tensor(X, dtype=torch.float32)
    
    predictions = []
    uncertainties = []
    
    model_wrapper.model.eval()
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            src, frac = batch_X.squeeze(-1).chunk(2, dim=1)
            
            src = src.to(device, dtype=torch.long)
            frac = frac.to(device, dtype=torch.float32)
            
            output = model_wrapper.model.forward(src, frac)
            pred, uncert = output.chunk(2, dim=-1)
            
            # Unscale predictions
            pred = model_wrapper.scaler.unscale(pred)
            uncert = torch.exp(uncert) * model_wrapper.scaler.std
            
            predictions.append(pred.cpu().numpy().flatten())
            uncertainties.append(uncert.cpu().numpy().flatten())
    
    predictions = np.concatenate(predictions)
    uncertainties = np.concatenate(uncertainties)
    
    if return_uncertainty:
        return predictions, uncertainties
    return predictions


def predict_from_csv(input_path, output_path, model_path=None):
    """
    Make predictions on a CSV file and save results.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file (must have 'formula' column with SMILES)
    output_path : str
        Path to save predictions
    model_path : str, optional
        Path to model file
    """
    # Load data
    df = pd.read_csv(input_path)
    
    if 'formula' not in df.columns:
        raise ValueError("Input CSV must have a 'formula' column with SMILES strings")
    
    smiles_list = df['formula'].tolist()
    
    # Load model
    if model_path:
        wrapper = load_trained_model(model_path)
    else:
        wrapper = load_trained_model()
    
    # Make predictions
    print(f"Making predictions on {len(smiles_list)} samples...")
    predictions, uncertainties = predict_tg(smiles_list, wrapper)
    
    # Create output DataFrame
    df_out = df.copy()
    df_out['predicted_tg'] = predictions
    df_out['uncertainty'] = uncertainties
    
    # Save
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Mean predicted Tg: {predictions.mean():.1f} °C")
    print(f"  Std predicted Tg:  {predictions.std():.1f} °C")
    print(f"  Min predicted Tg:  {predictions.min():.1f} °C")
    print(f"  Max predicted Tg:  {predictions.max():.1f} °C")
    
    return df_out


def main():
    parser = argparse.ArgumentParser(
        description='Predict glass transition temperature from polymer SMILES'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Input CSV file with SMILES in "formula" column'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--model', '-m', type=str, default=None,
        help='Path to model file (default: models/trained_models/psmiles_tg.pth)'
    )
    
    args = parser.parse_args()
    
    predict_from_csv(args.input, args.output, args.model)


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) > 1:
        main()
    else:
        # Demo predictions
        print("PSMILESNet Glass Transition Temperature Predictor")
        print("=" * 50)
        
        # Example polymers
        test_smiles = [
            '*CC(*)c1ccccc1',      # Polystyrene
            '*CC(*)(C)C(=O)OC',    # PMMA
            '*CC(*)C',             # Polypropylene
            '*CC*',                # Polyethylene
            '*CC(*)OC',            # Polyethylene oxide derivative
        ]
        
        print("\nExample predictions:")
        print("-" * 50)
        
        predictions, uncertainties = predict_tg(test_smiles)
        
        for smiles, pred, unc in zip(test_smiles, predictions, uncertainties):
            print(f"  {smiles:30s} -> Tg = {pred:6.1f} ± {unc:5.1f} °C")
        
        print("\n" + "=" * 50)
        print("Usage: python predict_tg.py --input data.csv --output predictions.csv")
