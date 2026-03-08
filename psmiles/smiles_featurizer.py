"""
SMILES Featurizer for PSMILESNet (CrabNet adapted for Polymer SMILES)

This module provides utilities to convert SMILES strings into the format
expected by CrabNet: atom indices and fractional compositions.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import torch
from torch.utils.data import Dataset, DataLoader

# Atom symbols we care about for organic molecules/polymers
# Index 0 is reserved for padding
ATOM_SYMBOLS = [
    'PAD',      # 0 - padding token
    'C',        # 1 - Carbon
    'N',        # 2 - Nitrogen  
    'O',        # 3 - Oxygen
    'S',        # 4 - Sulfur
    'F',        # 5 - Fluorine
    'Cl',       # 6 - Chlorine
    'Br',       # 7 - Bromine
    'I',        # 8 - Iodine
    'P',        # 9 - Phosphorus
    'B',        # 10 - Boron
    'Si',       # 11 - Silicon
    'Se',       # 12 - Selenium
    'H',        # 13 - Hydrogen (explicit)
    '*',        # 14 - Wildcard/connection point in polymers
]

ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(ATOM_SYMBOLS)}
IDX_TO_ATOM = {idx: atom for idx, atom in enumerate(ATOM_SYMBOLS)}
NUM_ATOM_TYPES = len(ATOM_SYMBOLS)

# Atom features for embedding
# [atomic_number, mass, electronegativity, num_valence, atomic_radius, is_aromatic_common]
ATOM_FEATURES = {
    'PAD': [0, 0.0, 0.0, 0, 0.0, 0],
    'C':   [6, 12.01, 2.55, 4, 0.77, 1],
    'N':   [7, 14.01, 3.04, 5, 0.75, 1],
    'O':   [8, 16.00, 3.44, 6, 0.73, 1],
    'S':   [16, 32.07, 2.58, 6, 1.02, 1],
    'F':   [9, 19.00, 3.98, 7, 0.71, 0],
    'Cl':  [17, 35.45, 3.16, 7, 0.99, 0],
    'Br':  [35, 79.90, 2.96, 7, 1.14, 0],
    'I':   [53, 126.90, 2.66, 7, 1.33, 0],
    'P':   [15, 30.97, 2.19, 5, 1.06, 0],
    'B':   [5, 10.81, 2.04, 3, 0.82, 0],
    'Si':  [14, 28.09, 1.90, 4, 1.11, 0],
    'Se':  [34, 78.96, 2.55, 6, 1.16, 1],
    'H':   [1, 1.008, 2.20, 1, 0.32, 0],
    '*':   [0, 0.0, 0.0, 0, 0.0, 0],  # Connection point
}


def get_atom_feature_matrix():
    """
    Create a feature matrix for all atom types.
    Shape: (NUM_ATOM_TYPES, num_features)
    """
    features = []
    for atom in ATOM_SYMBOLS:
        features.append(ATOM_FEATURES[atom])
    return np.array(features, dtype=np.float32)


def parse_psmiles(psmiles):
    """
    Parse a polymer SMILES string and extract atom counts.
    
    The * character represents connection points in polymer repeat units.
    We handle this by converting * to a special token before RDKit parsing.
    
    Parameters
    ----------
    psmiles : str
        Polymer SMILES string with * as connection points
        
    Returns
    -------
    atom_counts : OrderedDict
        Dictionary mapping atom symbols to their counts, sorted by count (descending)
    """
    # Replace * with a placeholder that RDKit can handle
    # Use [At] (Astatine) as a placeholder for *
    smiles_clean = psmiles.replace('*', '[At]')
    
    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            # Try without sanitization for edge cases
            mol = Chem.MolFromSmiles(smiles_clean, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
    except:
        mol = None
    
    atom_counts = {}
    
    if mol is not None:
        # Add explicit hydrogens for accurate counting
        mol = Chem.AddHs(mol)
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            # Convert Astatine back to *
            if symbol == 'At':
                symbol = '*'
            if symbol in ATOM_TO_IDX:
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            else:
                # Unknown atom type - skip or map to closest
                pass
    else:
        # Fallback: simple regex-based parsing
        atom_counts = parse_psmiles_fallback(psmiles)
    
    # Sort by count (descending)
    atom_counts = OrderedDict(sorted(atom_counts.items(), key=lambda x: -x[1]))
    
    return atom_counts


def parse_psmiles_fallback(psmiles):
    """
    Fallback parser for SMILES that RDKit can't handle.
    Uses simple pattern matching.
    """
    import re
    
    atom_counts = {}
    
    # Pattern for atoms: uppercase letter optionally followed by lowercase
    # Handle special cases like Cl, Br, Si
    patterns = [
        (r'Cl', 'Cl'),
        (r'Br', 'Br'),
        (r'Si', 'Si'),
        (r'Se', 'Se'),
        (r'\*', '*'),
    ]
    
    # Process special patterns first
    temp_smiles = psmiles
    for pattern, symbol in patterns:
        matches = re.findall(pattern, temp_smiles)
        atom_counts[symbol] = atom_counts.get(symbol, 0) + len(matches)
        temp_smiles = re.sub(pattern, '', temp_smiles)
    
    # Count remaining single-letter atoms
    single_atoms = re.findall(r'[CNOFPBIH]', temp_smiles, re.IGNORECASE)
    for atom in single_atoms:
        atom = atom.upper()
        if atom in ATOM_TO_IDX:
            atom_counts[atom] = atom_counts.get(atom, 0) + 1
    
    # Sort by count (descending)
    atom_counts = OrderedDict(sorted(atom_counts.items(), key=lambda x: -x[1]))
    
    return atom_counts


def get_molecular_descriptors(psmiles):
    """
    Get additional molecular descriptors from RDKit.
    
    Returns
    -------
    descriptors : dict
        Dictionary of molecular descriptors
    """
    smiles_clean = psmiles.replace('*', '[At]')
    
    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return None
    except:
        return None
    
    try:
        descriptors = {
            'mol_weight': Descriptors.MolWt(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        }
    except:
        descriptors = None
    
    return descriptors


def smiles_to_edm(smiles_list, targets, n_elements=8, verbose=True, scale=True):
    """
    Convert a list of SMILES strings to the EDM format expected by CrabNet.
    
    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    targets : array-like
        Target values
    n_elements : int
        Maximum number of atom types to consider per molecule
    verbose : bool
        Whether to show progress bar
    scale : bool
        Whether to normalize fractions
        
    Returns
    -------
    out : np.ndarray
        Shape (n_samples, 2*n_elements, 1) - atom indices and fractions
    y : np.ndarray
        Target values
    formulas : np.ndarray
        Original SMILES strings
    """
    n_samples = len(smiles_list)
    y = np.array(targets, dtype=np.float32)
    formulas = np.array(smiles_list)
    
    # Arrays to store atom indices and fractions
    elem_num = np.zeros((n_samples, n_elements), dtype=np.float32)
    elem_frac = np.zeros((n_samples, n_elements), dtype=np.float32)
    
    iterator = enumerate(smiles_list)
    if verbose:
        iterator = enumerate(tqdm(smiles_list, desc="Parsing SMILES", unit="molecules"))
    
    for i, smiles in iterator:
        atom_counts = parse_psmiles(smiles)
        
        total_atoms = sum(atom_counts.values())
        if total_atoms == 0:
            total_atoms = 1  # Avoid division by zero
        
        for j, (atom, count) in enumerate(atom_counts.items()):
            if j >= n_elements:
                break
            
            if atom in ATOM_TO_IDX:
                elem_num[i, j] = ATOM_TO_IDX[atom]
                if scale:
                    elem_frac[i, j] = count / total_atoms
                else:
                    elem_frac[i, j] = count
    
    # Reshape to match CrabNet format
    elem_num = elem_num.reshape(n_samples, n_elements, 1)
    elem_frac = elem_frac.reshape(n_samples, n_elements, 1)
    out = np.concatenate((elem_num, elem_frac), axis=1)
    
    return out, y, formulas


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for SMILES data.
    """
    
    def __init__(self, data, n_comp):
        self.data = data
        self.n_comp = n_comp
        
        self.X = np.array(data[0])
        self.y = np.array(data[1])
        self.formula = np.array(data[2])  # Actually SMILES strings
        
        self.shape = [(self.X.shape), (self.y.shape), (self.formula.shape)]
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        formula = self.formula[idx]
        
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        
        return (X, y, formula)


class SMILESCsvLoader:
    """
    Data loader for SMILES CSV files.
    Analogous to EDM_CsvLoader but for SMILES data.
    """
    
    def __init__(self, csv_data, batch_size=64, n_elements=8,
                 inference=False, verbose=True, scale=True,
                 pin_memory=True, shuffle=True):
        
        self.csv_data = csv_data
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        
        # Load data
        if isinstance(csv_data, str):
            df = pd.read_csv(csv_data, keep_default_na=False, na_values=[''])
        else:
            df = csv_data
        
        # Get SMILES and targets
        smiles_list = df['formula'].values.tolist()
        targets = df['target'].values
        
        # Remove duplicates by averaging targets (unless inference)
        if not inference:
            df_temp = pd.DataFrame({'formula': smiles_list, 'target': targets})
            df_temp = df_temp.groupby('formula').mean().reset_index()
            smiles_list = df_temp['formula'].values.tolist()
            targets = df_temp['target'].values
        
        # Convert to EDM format
        self.main_data = list(smiles_to_edm(
            smiles_list, targets, n_elements=n_elements,
            verbose=verbose, scale=scale
        ))
        
        self.n_train = len(self.main_data[0])
        self.n_elements = n_elements
    
    def get_data_loaders(self, inference=False):
        """Get PyTorch DataLoader."""
        shuffle = not inference
        dataset = SMILESDataset(self.main_data, self.n_elements)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=shuffle
        )
        return loader


def create_atom_embeddings_csv(output_path='data/element_properties/psmiles_atoms.csv'):
    """
    Create a CSV file with atom embeddings for SMILES atoms.
    This is analogous to the mat2vec element embeddings used in CrabNet.
    """
    import os
    
    # Create feature matrix
    # Features: atomic_num, mass, electronegativity, valence, radius, is_aromatic
    # We'll expand these into a richer representation using sine/cosine encoding
    
    features = []
    for atom in ATOM_SYMBOLS:
        feat = ATOM_FEATURES[atom]
        # Normalize features
        atomic_num = feat[0] / 53  # Max is Iodine (53)
        mass = feat[1] / 127  # Max is Iodine mass
        electronegativity = feat[2] / 4  # Max ~4
        valence = feat[3] / 7  # Max 7
        radius = feat[4] / 1.4  # Max ~1.4
        is_aromatic = feat[5]
        
        # Create embedding with sine/cosine encodings and raw features
        embed = [
            atomic_num, mass, electronegativity, valence, radius, is_aromatic,
            np.sin(atomic_num * np.pi), np.cos(atomic_num * np.pi),
            np.sin(electronegativity * np.pi), np.cos(electronegativity * np.pi),
            np.sin(valence * np.pi / 3.5), np.cos(valence * np.pi / 3.5),
            np.sin(radius * np.pi), np.cos(radius * np.pi),
        ]
        features.append(embed)
    
    features = np.array(features, dtype=np.float32)
    
    # Create DataFrame
    col_names = [f'feat_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=col_names, index=ATOM_SYMBOLS)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved atom embeddings to {output_path}")
    
    return df


if __name__ == '__main__':
    # Test the parser
    test_smiles = [
        '*C*',
        '*CC(*)C',
        '*CC(*)c1ccccc1',
        '*CC(*)OC(=O)c1ccccc1',
    ]
    
    for smiles in test_smiles:
        counts = parse_psmiles(smiles)
        print(f"{smiles}: {dict(counts)}")
    
    # Create atom embeddings file
    create_atom_embeddings_csv()
