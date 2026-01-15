"""
CrabNet Predictor Module

Handles loading and running CrabNet models for property prediction.
"""
import os
import sys

# Add parent directory to path for CrabNet imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Optional

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device


class CrabNetPredictor:
    """
    Singleton class for managing CrabNet model loading and predictions.
    
    Caches loaded models to avoid reloading for repeated predictions.
    """
    
    def __init__(self):
        self.compute_device = get_compute_device(prefer_last=True)
        self._loaded_models = {}
        self._model_list = None
        self.is_initialized = True
        
        # Set working directory to parent for model loading
        self._parent_dir = parent_dir
        
        print(f"CrabNet Predictor initialized on device: {self.compute_device}")
    
    def list_models(self) -> List[str]:
        """
        List all available pre-trained models.
        
        Returns:
            List of model names (without .pth extension)
        """
        if self._model_list is not None:
            return self._model_list
        
        models_dir = os.path.join(self._parent_dir, 'models', 'trained_models')
        if not os.path.exists(models_dir):
            return []
        
        model_files = [
            f.replace('.pth', '') 
            for f in os.listdir(models_dir) 
            if f.endswith('.pth')
        ]
        
        self._model_list = sorted(model_files)
        return self._model_list
    
    def _load_model(self, model_name: str) -> Model:
        """
        Load a CrabNet model by name.
        
        Args:
            model_name: Name of the model (without .pth extension)
            
        Returns:
            Loaded Model instance
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        # Change to parent directory for proper path resolution
        original_cwd = os.getcwd()
        os.chdir(self._parent_dir)
        
        try:
            # Suppress verbose output
            model = Model(
                CrabNet(compute_device=self.compute_device).to(self.compute_device),
                model_name=model_name,
                verbose=False
            )
            
            # Load pre-trained weights
            model.load_network(f'{model_name}.pth')
            
            # Cache the model
            self._loaded_models[model_name] = model
            
            return model
        finally:
            os.chdir(original_cwd)
    
    def predict(self, formula: str, model_name: str) -> Tuple[float, float]:
        """
        Predict a property for a given formula.
        
        Args:
            formula: Chemical formula (normalized, e.g., "Fe2O3")
            model_name: Name of the property model
            
        Returns:
            Tuple of (predicted_value, uncertainty)
        """
        # Load model
        model = self._load_model(model_name)
        
        # Change to parent directory for data loading
        original_cwd = os.getcwd()
        os.chdir(self._parent_dir)
        
        try:
            # Create temporary DataFrame for prediction
            temp_df = pd.DataFrame({
                'formula': [formula],
                'target': [0.0]  # Dummy target
            })
            
            # Save to temporary CSV (required by CrabNet's data loader)
            temp_csv = os.path.join(self._parent_dir, '_temp_predict.csv')
            temp_df.to_csv(temp_csv, index=False)
            
            try:
                # Load data and predict
                model.load_data(temp_csv, batch_size=1, train=False)
                act, pred, formulae, uncertainty = model.predict(model.data_loader)
                
                return float(pred[0]), float(uncertainty[0])
            finally:
                # Clean up temp file
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
        finally:
            os.chdir(original_cwd)
    
    def predict_batch(self, formulas: List[str], model_name: str) -> List[Tuple[float, float]]:
        """
        Predict properties for multiple formulas.
        
        Args:
            formulas: List of chemical formulas
            model_name: Name of the property model
            
        Returns:
            List of (predicted_value, uncertainty) tuples
        """
        # Load model
        model = self._load_model(model_name)
        
        # Change to parent directory for data loading
        original_cwd = os.getcwd()
        os.chdir(self._parent_dir)
        
        try:
            # Create temporary DataFrame
            temp_df = pd.DataFrame({
                'formula': formulas,
                'target': [0.0] * len(formulas)
            })
            
            # Save to temporary CSV
            temp_csv = os.path.join(self._parent_dir, '_temp_predict_batch.csv')
            temp_df.to_csv(temp_csv, index=False)
            
            try:
                # Load data and predict
                batch_size = min(512, len(formulas))
                model.load_data(temp_csv, batch_size=batch_size, train=False)
                act, pred, formulae, uncertainty = model.predict(model.data_loader)
                
                return [(float(p), float(u)) for p, u in zip(pred, uncertainty)]
            finally:
                # Clean up temp file
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
        finally:
            os.chdir(original_cwd)
    
    def clear_cache(self):
        """Clear cached models to free memory."""
        self._loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Singleton instance
_predictor_instance = None

def get_predictor() -> CrabNetPredictor:
    """Get the singleton CrabNet predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CrabNetPredictor()
    return _predictor_instance
