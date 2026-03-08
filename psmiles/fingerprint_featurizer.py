"""
Fingerprint-based featurization for polymer SMILES.

Uses Morgan fingerprints (ECFP) and RDKit molecular descriptors
to capture structural information from SMILES, which is much more
informative than simple atom counts.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs

# RDKit descriptors to compute
DESCRIPTOR_NAMES = [
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

def smiles_to_mol(smiles):
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


def get_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Compute Morgan fingerprint (ECFP) for a molecule.
    
    Parameters
    ----------
    mol : RDKit molecule
    radius : int
        Radius of the fingerprint (2 = ECFP4, 3 = ECFP6)
    n_bits : int
        Number of bits in the fingerprint
        
    Returns
    -------
    fp : np.ndarray
        Binary fingerprint array
    """
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits, dtype=np.float32)


def get_molecular_descriptors(mol):
    """
    Compute molecular descriptors using RDKit.
    
    Returns
    -------
    descriptors : np.ndarray
        Array of descriptor values
    """
    if mol is None:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
    
    desc_values = []
    for name in DESCRIPTOR_NAMES:
        try:
            if hasattr(Descriptors, name):
                func = getattr(Descriptors, name)
                value = func(mol)
            elif hasattr(rdMolDescriptors, f'Calc{name}'):
                func = getattr(rdMolDescriptors, f'Calc{name}')
                value = func(mol)
            else:
                value = 0.0
            
            # Handle NaN/Inf
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            desc_values.append(float(value))
        except:
            desc_values.append(0.0)
    
    return np.array(desc_values, dtype=np.float32)


def featurize_smiles(smiles, fp_radius=2, fp_bits=2048):
    """
    Featurize a SMILES string using fingerprints and descriptors.
    
    Returns
    -------
    features : np.ndarray
        Concatenated fingerprint and descriptors
    """
    mol = smiles_to_mol(smiles)
    
    # Get fingerprint
    fp = get_morgan_fingerprint(mol, radius=fp_radius, n_bits=fp_bits)
    
    # Get descriptors
    desc = get_molecular_descriptors(mol)
    
    # Concatenate
    features = np.concatenate([fp, desc])
    
    return features


def featurize_dataset(smiles_list, targets, fp_radius=2, fp_bits=2048, verbose=True):
    """
    Featurize a list of SMILES strings.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values
    formulas : np.ndarray
        Original SMILES strings
    """
    n_samples = len(smiles_list)
    n_features = fp_bits + len(DESCRIPTOR_NAMES)
    
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    formulas = np.array(smiles_list)
    
    iterator = enumerate(smiles_list)
    if verbose:
        iterator = enumerate(tqdm(smiles_list, desc="Featurizing SMILES"))
    
    for i, smiles in iterator:
        X[i] = featurize_smiles(smiles, fp_radius, fp_bits)
    
    return X, y, formulas


class FingerprintDataset(Dataset):
    """PyTorch Dataset for fingerprint-based features."""
    
    def __init__(self, X, y, formulas):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.formulas = formulas
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.formulas[idx]


class FingerprintCsvLoader:
    """
    Data loader for SMILES CSV files using fingerprint featurization.
    """
    
    def __init__(self, csv_data, batch_size=64, fp_radius=2, fp_bits=2048,
                 inference=False, verbose=True, pin_memory=True, shuffle=True):
        
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.fp_bits = fp_bits
        self.n_descriptors = len(DESCRIPTOR_NAMES)
        
        # Load data
        if isinstance(csv_data, str):
            df = pd.read_csv(csv_data, keep_default_na=False, na_values=[''])
        else:
            df = csv_data
        
        smiles_list = df['formula'].values.tolist()
        targets = df['target'].values
        
        # Remove duplicates by averaging (unless inference)
        if not inference:
            df_temp = pd.DataFrame({'formula': smiles_list, 'target': targets})
            df_temp = df_temp.groupby('formula').mean().reset_index()
            smiles_list = df_temp['formula'].values.tolist()
            targets = df_temp['target'].values
        
        # Featurize
        self.X, self.y, self.formulas = featurize_dataset(
            smiles_list, targets, fp_radius=fp_radius, 
            fp_bits=fp_bits, verbose=verbose
        )
        
        self.n_samples = len(self.y)
        self.n_features = self.X.shape[1]
        
        if verbose:
            print(f"Features: {fp_bits} fingerprint bits + {self.n_descriptors} descriptors = {self.n_features} total")
    
    def get_data_loaders(self, inference=False):
        """Get PyTorch DataLoader."""
        shuffle = not inference
        dataset = FingerprintDataset(self.X, self.y, self.formulas)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            num_workers=0
        )
        return loader
    
    def get_feature_stats(self):
        """Get feature statistics for normalization."""
        # Compute mean and std, avoiding division by zero
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return mean, std


if __name__ == '__main__':
    # Test featurization
    test_smiles = [
        '*CC(*)c1ccccc1',
        '*CC(*)(C)C(=O)OC',
        '*CC(*)OC(=O)c1ccccc1',
    ]
    
    for smiles in test_smiles:
        features = featurize_smiles(smiles)
        print(f"{smiles}: {features.shape} features, {features.sum():.0f} active bits")
