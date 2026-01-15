"""
Simple interactive script to predict material properties using CrabNet

Usage:
    python simple_predict.py
"""
import os
import sys

# Add the repository root to Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run_inference import (
    load_pretrained_model, 
    predict_from_formulas, 
    predict_from_csv,
    list_available_models
)

def main():
    print("=" * 70)
    print("CrabNet - Simple Material Property Predictor")
    print("=" * 70)
    
    # Show available models
    print("\n📋 Available Properties to Predict:\n")
    models = list_available_models()
    
    # Show key models organized by category
    print("Band Gap:")
    print("  1. OQMD_Bandgap")
    print("  2. aflow__Egap")
    
    print("\nFormation Energy/Enthalpy:")
    print("  3. OQMD_Formation_Enthalpy")
    print("  4. OQMD_Energy_per_atom")
    
    print("\nMechanical Properties:")
    print("  5. aflow__ael_bulk_modulus_vrh (Bulk Modulus)")
    print("  6. aflow__ael_shear_modulus_vrh (Shear Modulus)")
    print("  7. mp_elastic_anisotropy (Elastic Anisotropy)")
    
    print("\nThermal Properties:")
    print("  8. aflow__agl_thermal_conductivity_300K")
    print("  9. aflow__agl_thermal_expansion_300K")
    print(" 10. aflow__ael_debye_temperature")
    
    print(f"\n... and {len(models) - 10} more models available")
    print("\nTo see all models, check INFERENCE_GUIDE.md")
    
    print("\n" + "=" * 70)
    
    # Get user input
    print("\nWhat would you like to predict?")
    model_name = input("Enter model name (e.g., OQMD_Bandgap): ").strip()
    
    if not model_name:
        model_name = "OQMD_Bandgap"
        print(f"Using default: {model_name}")
    
    if model_name not in models:
        print(f"⚠ Warning: Model '{model_name}' not found in trained_models/")
        print(f"Available models: {', '.join(models[:5])}...")
        return
    
    # Load model
    print(f"\nLoading model: {model_name}...")
    try:
        model = load_pretrained_model(model_name, verbose=False)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Get formulas
    print("\n" + "=" * 70)
    print("Enter chemical formulas to predict")
    print("Examples: Fe2O3, TiO2, Al2O3, SiO2, GaN, ZnO")
    print("=" * 70)
    
    formulas_input = input("\nEnter formulas (comma-separated): ").strip()
    
    if not formulas_input:
        # Use default examples
        formulas = ['Fe2O3', 'TiO2', 'Al2O3', 'SiO2']
        print(f"Using example formulas: {', '.join(formulas)}")
    else:
        formulas = [f.strip() for f in formulas_input.split(',')]
    
    # Run predictions
    print(f"\n🔮 Predicting {model_name} for {len(formulas)} materials...")
    print("-" * 70)
    
    try:
        results = predict_from_formulas(model, formulas)
        
        # Display results
        print("\n📊 Results:\n")
        for idx, row in results.iterrows():
            formula = row['formula']
            pred = row['predicted']
            uncert = row['uncertainty']
            print(f"  {formula:15s} → {pred:12.6f} ± {uncert:.6f}")
        
        # Save results
        output_file = f"predictions_{model_name}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Done!")
    print("\nTips:")
    print("- Lower uncertainty = more confident prediction")
    print("- Try different models for different properties")
    print("- See INFERENCE_GUIDE.md for more examples")
    print("=" * 70)


if __name__ == '__main__':
    main()
