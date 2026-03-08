"""
PSMILES (Polymer SMILES) Predictor Module

Handles loading and running BioCrabNet models for polymer property prediction.
Supports V1 (PSMILESNet), V2 (FingerprintNet single), and V3 (FingerprintNet ensemble).
"""
import os
import sys
import io
import base64
from typing import Tuple, List, Dict, Any, Optional

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

# Import PSMILESNet for V1
try:
    from psmiles.psmiles_model import PSMILESNet
    from psmiles.smiles_featurizer import smiles_to_edm
    from utils.utils import Scaler
    PSMILES_AVAILABLE = True
except ImportError:
    PSMILES_AVAILABLE = False
    print("Warning: PSMILESNet (V1) not available. V2/V3 will still work.")

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# Descriptor Configuration
# ============================================================================

# 9 well-known and relevant descriptors for polymer Tg
DISPLAY_DESCRIPTORS = [
    ('MolWt', 'Molecular Weight', 'g/mol'),
    ('NumRotatableBonds', 'Rotatable Bonds', 'count'),
    ('NumAromaticRings', 'Aromatic Rings', 'count'),
    ('TPSA', 'Polar Surface Area', 'Å²'),
    ('FractionCSP3', 'Fraction sp³ Carbons', 'ratio'),
    ('NumHBondDonors', 'H-Bond Donors', 'count'),
    ('NumHBondAcceptors', 'H-Bond Acceptors', 'count'),
    ('RingCount', 'Total Rings', 'count'),
    ('HeavyAtomCount', 'Heavy Atoms', 'count'),
]

# All descriptors used for fingerprint model prediction
ALL_DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA',
    'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors',
    'NumHeteroatoms', 'NumValenceElectrons',
    'FractionCSP3', 'NumAromaticRings', 'NumAliphaticRings',
    'NumSaturatedRings', 'NumAromaticHeterocycles', 
    'NumAromaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticCarbocycles', 'RingCount',
    'HeavyAtomCount', 'NHOHCount', 'NOCount',
    'NumRadicalElectrons', 'LabuteASA',
    'BalabanJ', 'BertzCT',
]


# ============================================================================
# SMILES Utilities
# ============================================================================

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert polymer SMILES to RDKit molecule.
    Handles the * connection points by replacing with methyl groups.
    """
    # Replace * with methyl groups for RDKit processing
    smiles_clean = smiles.replace('*', '[CH3]')
    
    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            # Try with different replacement
            smiles_clean = smiles.replace('*', 'C')
            mol = Chem.MolFromSmiles(smiles_clean)
    except:
        mol = None
    
    return mol


def get_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint (ECFP) for a molecule."""
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits, dtype=np.float32)


def get_molecular_descriptors(mol) -> np.ndarray:
    """Compute all molecular descriptors for prediction."""
    if mol is None:
        return np.zeros(len(ALL_DESCRIPTOR_NAMES), dtype=np.float32)
    
    desc_values = []
    for name in ALL_DESCRIPTOR_NAMES:
        try:
            if hasattr(Descriptors, name):
                func = getattr(Descriptors, name)
                value = func(mol)
            elif hasattr(rdMolDescriptors, f'Calc{name}'):
                func = getattr(rdMolDescriptors, f'Calc{name}')
                value = func(mol)
            else:
                value = 0.0
            
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            desc_values.append(float(value))
        except:
            desc_values.append(0.0)
    
    return np.array(desc_values, dtype=np.float32)


def get_display_descriptors(smiles: str) -> List[Dict[str, Any]]:
    """
    Get the 9 display descriptors for a polymer SMILES.
    
    Returns a list of dicts with name, value, description, units.
    """
    mol = smiles_to_mol(smiles)
    
    results = []
    for name, description, units in DISPLAY_DESCRIPTORS:
        value = None
        try:
            if mol is not None:
                if name == 'NumHBondDonors':
                    value = Descriptors.NumHDonors(mol)
                elif name == 'NumHBondAcceptors':
                    value = Descriptors.NumHAcceptors(mol)
                elif hasattr(Descriptors, name):
                    value = getattr(Descriptors, name)(mol)
                
                if value is not None and (np.isnan(value) or np.isinf(value)):
                    value = None
        except:
            value = None
        
        results.append({
            'name': name,
            'description': description,
            'value': round(value, 4) if value is not None else None,
            'units': units
        })
    
    return results


def featurize_smiles(smiles: str, fp_radius: int = 2, fp_bits: int = 2048) -> np.ndarray:
    """Featurize a SMILES string using fingerprints and descriptors."""
    mol = smiles_to_mol(smiles)
    fp = get_morgan_fingerprint(mol, radius=fp_radius, n_bits=fp_bits)
    desc = get_molecular_descriptors(mol)
    return np.concatenate([fp, desc])


def generate_structure_image(smiles: str, size: Tuple[int, int] = (400, 400)) -> str:
    """
    Generate a structure image for a polymer SMILES.
    
    Returns base64-encoded PNG image data.
    """
    mol = smiles_to_mol(smiles)
    
    if mol is None:
        # Return empty/error image
        return ""
    
    try:
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw molecule
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error generating structure image: {e}")
        return ""


# ============================================================================
# Model Loading - FingerprintNet (V2/V3)
# ============================================================================

class ResidualBlock(torch.nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.fc2 = torch.nn.Linear(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.act = torch.nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = x + residual
        x = self.act(x)
        return x


class FingerprintNet(torch.nn.Module):
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
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.LeakyReLU(0.1))
            layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.input_layers = torch.nn.Sequential(*layers)
        
        # Residual blocks
        self.residual_blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_dims[-1], dropout=dropout)
            for _ in range(n_residual_blocks)
        ])
        
        # Output head - predicts mean and log_std
        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 2)  # [mean, log_std]
        )
    
    def forward(self, x):
        x = self.input_layers(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        output = self.output_head(x)
        return output


# ============================================================================
# PSMILES Predictor Class
# ============================================================================

class PSMILESPredictor:
    """
    Predictor class for polymer SMILES property prediction.
    
    Supports three model versions:
    - V1: PSMILESNet (transformer-based, atom counts)
    - V2: FingerprintNet (single model, fingerprints)
    - V3: FingerprintNet ensemble (5 models averaged)
    """
    
    AVAILABLE_MODELS = {
        'v1': {
            'name': 'PSMILESNet (V1)',
            'description': 'Transformer-based model using atom counts',
            'file': 'psmiles_tg.pth'
        },
        'v2': {
            'name': 'FingerprintNet (V2)',
            'description': 'DNN with Morgan fingerprints',
            'file': 'psmiles_tg_fingerprint.pth'
        },
        'v3': {
            'name': 'FingerprintNet Ensemble (V3)',
            'description': '5-fold cross-validation ensemble (most accurate)',
            'file': 'psmiles_tg_ensemble.pth'
        }
    }
    
    PROPERTIES = {
        'glass_transition_temperature': {
            'name': 'Glass Transition Temperature',
            'description': 'Temperature at which polymer transitions from glassy to rubbery state',
            'units': '°C'
        }
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._loaded_models = {}
        self._parent_dir = parent_dir
        
        print(f"PSMILES Predictor initialized on device: {self.device}")
    
    def list_models(self) -> List[Dict[str, str]]:
        """List available model versions."""
        return [
            {'id': k, **v} for k, v in self.AVAILABLE_MODELS.items()
        ]
    
    def list_properties(self) -> List[Dict[str, str]]:
        """List available properties to predict."""
        return [
            {'id': k, **v} for k, v in self.PROPERTIES.items()
        ]
    
    def _load_v1_model(self, model_path: str) -> Tuple[PSMILESNet, Any, Any]:
        """Load a V1 PSMILESNet model."""
        if not PSMILES_AVAILABLE:
            raise ImportError("PSMILESNet (V1) dependencies not available")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config
        config = checkpoint.get('model_config', {})
        
        # Create model
        model = PSMILESNet(
            out_dims=config.get('out_dims', 3),
            d_model=config.get('d_model', 256),
            N=config.get('N', 3),
            heads=config.get('heads', 4),
            compute_device=self.device
        )
        model.load_state_dict(checkpoint['weights'])
        model = model.to(self.device)
        model.eval()
        
        # V1 doesn't use separate X and y scalers - only has one scaler for y
        # Return None for scaler_X since V1 doesn't scale inputs
        return model, None, checkpoint['scaler_state']
    
    def _load_v2_model(self, model_path: str) -> Tuple[FingerprintNet, Any, Any]:
        """Load a V2 FingerprintNet model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint['model_config']
        model = FingerprintNet(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            n_residual_blocks=config['n_residual_blocks'],
            dropout=config.get('dropout', 0.15)
        )
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(self.device)
        model.eval()
        
        return model, checkpoint['scaler_X'], checkpoint['scaler_y']
    
    def _load_v3_ensemble(self, model_path: str) -> Tuple[List[FingerprintNet], List[Any], List[Any]]:
        """Load a V3 ensemble model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint['model_config']
        models = []
        
        for state_dict in checkpoint['models_state']:
            model = FingerprintNet(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims'],
                n_residual_blocks=config['n_residual_blocks'],
                dropout=config.get('dropout', 0.15)
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            models.append(model)
        
        return models, checkpoint['scalers_X'], checkpoint['scalers_y']
    
    def _get_model(self, version: str):
        """Get a loaded model (with caching)."""
        if version in self._loaded_models:
            return self._loaded_models[version]
        
        model_info = self.AVAILABLE_MODELS.get(version)
        if not model_info:
            raise ValueError(f"Unknown model version: {version}")
        
        model_path = os.path.join(
            self._parent_dir, 'models', 'trained_models', model_info['file']
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if version == 'v1':
            # V1 uses transformer-based PSMILESNet architecture
            if not PSMILES_AVAILABLE:
                raise ImportError(
                    "V1 (PSMILESNet) dependencies not available. "
                    "Please ensure psmiles module is properly installed."
                )
            model_data = self._load_v1_model(model_path)
            self._loaded_models[version] = ('v1', *model_data)
        elif version == 'v2':
            model_data = self._load_v2_model(model_path)
            self._loaded_models[version] = ('single', *model_data)
        elif version == 'v3':
            model_data = self._load_v3_ensemble(model_path)
            self._loaded_models[version] = ('ensemble', *model_data)
        
        return self._loaded_models[version]
    
    def predict(self, smiles: str, model_version: str = 'v3') -> Tuple[float, float]:
        """
        Predict glass transition temperature for a polymer SMILES.
        
        Args:
            smiles: Polymer SMILES string
            model_version: 'v1', 'v2', or 'v3'
            
        Returns:
            Tuple of (predicted_value, uncertainty)
        """
        # Get model
        model_data = self._get_model(model_version)
        model_type = model_data[0]
        
        # V1 uses different featurization (atom sequences)
        if model_type == 'v1':
            model, _, scaler_state = model_data[1], model_data[2], model_data[3]
            
            # Reconstruct scaler from state dict
            scaler = Scaler(torch.zeros(3))
            scaler.load_state_dict(scaler_state)
            
            # Convert SMILES to EDM format (requires dummy targets)
            dummy_targets = np.zeros(1)
            X, _, formulas = smiles_to_edm([smiles], dummy_targets, n_elements=10, verbose=False, scale=True)
            
            # X has shape (1, 2*n_elements, 1) - first half is atom indices, second half is fractions
            X = X[0, :, 0]  # Shape: (2*n_elements,)
            n_elements = len(X) // 2
            src = X[:n_elements]  # Atom indices
            frac = X[n_elements:]  # Fractions
            
            # Convert to tensors
            src_tensor = torch.as_tensor(src, dtype=torch.long).unsqueeze(0).to(self.device)
            frac_tensor = torch.as_tensor(frac, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = model(src_tensor, frac_tensor)
                pred_scaled = output[0, 0].cpu()
                log_std = output[0, 1].cpu()
                uncert_scaled = torch.exp(log_std)
            
            # Unscale predictions
            pred = scaler.unscale(pred_scaled).item()
            uncert = (uncert_scaled * scaler.std).item()
            
            return float(pred), float(uncert)
        
        # V2/V3 use fingerprint featurization
        features = featurize_smiles(smiles)
        
        if model_type == 'single':
            model, scaler_X, scaler_y = model_data[1], model_data[2], model_data[3]
            
            # Scale features
            X_scaled = scaler_X.transform(features.reshape(1, -1))
            X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = model(X_tensor)
                pred = output[0, 0].cpu().numpy()
                log_std = output[0, 1].cpu().numpy()
                uncert = np.exp(log_std)
            
            # Unscale
            pred = scaler_y.inverse_transform([[pred]])[0, 0]
            uncert = uncert * scaler_y.scale_[0]
            
        elif model_type == 'ensemble':
            models, scalers_X, scalers_y = model_data[1], model_data[2], model_data[3]
            
            all_preds = []
            all_uncerts = []
            
            for model, scaler_X, scaler_y in zip(models, scalers_X, scalers_y):
                X_scaled = scaler_X.transform(features.reshape(1, -1))
                X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    output = model(X_tensor)
                    p = output[0, 0].cpu().numpy()
                    ls = output[0, 1].cpu().numpy()
                    u = np.exp(ls)
                
                p = scaler_y.inverse_transform([[p]])[0, 0]
                u = u * scaler_y.scale_[0]
                
                all_preds.append(p)
                all_uncerts.append(u)
            
            # Average predictions
            pred = np.mean(all_preds)
            # Combined uncertainty
            uncert = np.sqrt(np.mean(np.array(all_uncerts)**2) + np.var(all_preds))
        
        return float(pred), float(uncert)
    
    def get_structure_image(self, smiles: str) -> str:
        """Get base64-encoded structure image."""
        return generate_structure_image(smiles)
    
    def get_descriptors(self, smiles: str) -> List[Dict[str, Any]]:
        """Get display descriptors for a polymer."""
        return get_display_descriptors(smiles)
    
    def validate_smiles(self, smiles: str) -> Tuple[bool, str]:
        """
        Validate a polymer SMILES string.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not smiles.strip():
            return False, "SMILES string is empty"
        
        mol = smiles_to_mol(smiles)
        if mol is None:
            return False, "Invalid SMILES: could not parse structure"
        
        return True, ""


# Singleton instance
_predictor_instance = None

def get_psmiles_predictor() -> PSMILESPredictor:
    """Get the singleton PSMILES predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PSMILESPredictor()
    return _predictor_instance
