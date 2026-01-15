"""
Simple script to run CrabNet inference using pre-trained models
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

# Add the repository root to Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

# Set random seed for reproducibility
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Get compute device (GPU if available, otherwise CPU)
compute_device = get_compute_device(prefer_last=True)
print(f"Using device: {compute_device}")


def load_pretrained_model(model_name, verbose=True):
    """
    Load a pre-trained CrabNet model
    
    Parameters:
    -----------
    model_name : str
        Name of the model (without .pth extension)
        Available models are in models/trained_models/
    verbose : bool
        Print model information
    
    Returns:
    --------
    model : Model
        Loaded CrabNet model ready for inference
    """
    # Create model instance
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=model_name, 
                  verbose=verbose)
    
    # Load the pre-trained weights
    model_path = f'{model_name}.pth'
    if not os.path.exists(f'models/trained_models/{model_path}'):
        raise FileNotFoundError(f"Model file not found: models/trained_models/{model_path}")
    
    model.load_network(model_path)
    print(f"✓ Successfully loaded model: {model_name}")
    
    return model


def predict_from_csv(model, csv_file, batch_size=512):
    """
    Run predictions on a CSV file containing chemical formulas
    
    Parameters:
    -----------
    model : Model
        Loaded CrabNet model
    csv_file : str
        Path to CSV file with 'formula' and 'target' columns
        (target can be dummy values like 0 if unknown)
    batch_size : int
        Batch size for inference
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame with compositions, predictions, and uncertainties
    """
    # Load data
    model.load_data(csv_file, batch_size=batch_size, train=False)
    
    # Run prediction
    print(f"Running predictions on {len(model.data_loader.dataset)} compositions...")
    act, pred, formulae, uncertainty = model.predict(model.data_loader)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'formula': formulae,
        'actual': act,
        'predicted': pred,
        'uncertainty': uncertainty
    })
    
    return results_df


def predict_from_formulas(model, formulas, batch_size=512):
    """
    Run predictions on a list of chemical formulas
    
    Parameters:
    -----------
    model : Model
        Loaded CrabNet model
    formulas : list of str
        List of chemical formulas (e.g., ['Fe2O3', 'SiO2', 'Al2O3'])
    batch_size : int
        Batch size for inference
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame with compositions, predictions, and uncertainties
    """
    # Create temporary CSV file
    temp_csv = 'temp_inference_data.csv'
    temp_df = pd.DataFrame({
        'formula': formulas,
        'target': [0] * len(formulas)  # dummy targets
    })
    temp_df.to_csv(temp_csv, index=False)
    
    # Run prediction
    results_df = predict_from_csv(model, temp_csv, batch_size)
    
    # Clean up temp file
    os.remove(temp_csv)
    
    # Remove dummy actual values
    results_df = results_df.drop('actual', axis=1)
    
    return results_df


def list_available_models():
    """
    List all available pre-trained models
    
    Returns:
    --------
    models : list of str
        List of model names (without .pth extension)
    """
    models_dir = 'models/trained_models'
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return []
    
    model_files = [f.replace('.pth', '') for f in os.listdir(models_dir) 
                   if f.endswith('.pth') and f != 'README.md']
    return sorted(model_files)


# %%
if __name__ == '__main__':
    print("=" * 70)
    print("CrabNet Inference Script")
    print("=" * 70)
    
    # List available models
    print("\n📋 Available pre-trained models:")
    available_models = list_available_models()
    
    # Group models by property type for better display
    print(f"\nFound {len(available_models)} pre-trained models")
    print("\nSome example models:")
    for i, model_name in enumerate(available_models[:10]):
        print(f"  • {model_name}")
    if len(available_models) > 10:
        print(f"  ... and {len(available_models) - 10} more")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Predict from a list of formulas")
    print("=" * 70)
    
    # Example 1: Predict properties for specific formulas
    # Let's use the band gap model (OQMD_Bandgap)
    model_name = 'OQMD_Bandgap'
    
    print(f"\nLoading model: {model_name}")
    model = load_pretrained_model(model_name, verbose=False)
    
    # Define some test formulas
    test_formulas = [
        'Fe2O3',      # Iron oxide (Hematite)
        'SiO2',       # Silicon dioxide (Quartz)
        'Al2O3',      # Aluminum oxide (Corundum)
        'TiO2',       # Titanium dioxide
        'Cu2O',       # Copper(I) oxide
        'GaN',        # Gallium nitride
        'ZnO',        # Zinc oxide
    ]
    
    print(f"\nPredicting band gaps for {len(test_formulas)} materials...")
    results = predict_from_formulas(model, test_formulas)
    
    print("\n📊 Results:")
    print(results.to_string(index=False))
    
    # Save results to CSV
    output_file = 'inference_results_bandgap.csv'
    results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Predict from an existing CSV file")
    print("=" * 70)
    
    # Example 2: Predict using an existing CSV file
    csv_file = 'data/materials_data/example_materials_property/test.csv'
    
    if os.path.exists(csv_file):
        print(f"\nUsing example data from: {csv_file}")
        
        # For this we'll use the same model
        results_csv = predict_from_csv(model, csv_file)
        
        # Calculate MAE if actual values are available
        mae = np.abs(results_csv['actual'] - results_csv['predicted']).mean()
        print(f"\n📈 Mean Absolute Error: {mae:.4f}")
        
        # Show first few predictions
        print("\n📊 Sample results (first 10):")
        print(results_csv.head(10).to_string(index=False))
        
        # Save results
        output_file_csv = 'inference_results_from_csv.csv'
        results_csv.to_csv(output_file_csv, index=False)
        print(f"\n✓ Results saved to: {output_file_csv}")
    else:
        print(f"\nExample CSV file not found: {csv_file}")
    
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Try different property models")
    print("=" * 70)
    
    # Example 3: Try predicting different properties for the same material
    formula = 'Fe2O3'
    print(f"\nPredicting multiple properties for: {formula}\n")
    
    # Select a few different property models to try
    properties_to_test = [
        'OQMD_Bandgap',
        'OQMD_Formation_Enthalpy',
        'aflow__ael_bulk_modulus_vrh',
        'aflow__ael_shear_modulus_vrh',
    ]
    
    multi_property_results = []
    for prop_model in properties_to_test:
        if prop_model in available_models:
            try:
                print(f"  Loading {prop_model}...")
                model_prop = load_pretrained_model(prop_model, verbose=False)
                result = predict_from_formulas(model_prop, [formula])
                
                multi_property_results.append({
                    'formula': formula,
                    'property': prop_model,
                    'predicted_value': result['predicted'].values[0],
                    'uncertainty': result['uncertainty'].values[0]
                })
            except Exception as e:
                print(f"    ⚠ Error loading {prop_model}: {e}")
    
    if multi_property_results:
        multi_df = pd.DataFrame(multi_property_results)
        print("\n📊 Multi-property predictions:")
        print(multi_df.to_string(index=False))
        
        output_file_multi = 'inference_results_multi_property.csv'
        multi_df.to_csv(output_file_multi, index=False)
        print(f"\n✓ Results saved to: {output_file_multi}")
    
    
    print("\n" + "=" * 70)
    print("✅ Inference complete!")
    print("=" * 70)
    print("\nTo use this script for your own predictions:")
    print("1. Choose a model from the available models list")
    print("2. Prepare your formulas as a list or CSV file")
    print("3. Use predict_from_formulas() or predict_from_csv()")
    print("=" * 70)
