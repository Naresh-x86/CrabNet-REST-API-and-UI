"""
Chemical Formula Utilities

Functions for parsing, normalizing, and validating chemical formulas
for use with CrabNet.
"""
import re
from typing import Dict, List, Tuple
from collections import OrderedDict


# Element symbols (sorted by length for proper matching)
ELEMENTS = [
    'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'Ba', 'Be', 'Bh', 'Bi',
    'Bk', 'Br', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs',
    'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu', 'Fe', 'Fl', 'Fm', 'Fr', 'Ga',
    'Gd', 'Ge', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'In', 'Ir', 'Kr', 'La', 'Li',
    'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'Na', 'Nb', 'Nd',
    'Ne', 'Nh', 'Ni', 'No', 'Np', 'Og', 'Os', 'Pa', 'Pb', 'Pd', 'Pm', 'Po',
    'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru', 'Sb',
    'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th',
    'Ti', 'Tl', 'Tm', 'Ts', 'Uue', 'Ubn', 'Ubu', 'Ube', 'Ubt', 'Ubq', 'Ubp',
    'Ubh', 'Ubs', 'Ubo', 'Uen', 'Ueu', 'Tn', 'Xe', 'Yb', 'Zn', 'Zr',
    'B', 'C', 'F', 'H', 'I', 'K', 'N', 'O', 'P', 'S', 'U', 'V', 'W', 'Y'
]

# Create regex pattern - match longer symbols first
ELEMENT_PATTERN = re.compile(
    r'(' + '|'.join(sorted(ELEMENTS, key=len, reverse=True)) + r')(\d*\.?\d*)'
)


def parse_formula(formula: str) -> Dict[str, float]:
    """
    Parse a chemical formula into element counts.
    
    Handles:
    - Simple formulas: Fe2O3, NaCl, H2O
    - Without numbers: AlLiSi -> Al1Li1Si1
    - With parentheses: Ca(OH)2
    - Decimal coefficients: Fe0.5Ni0.5
    
    Args:
        formula: Chemical formula string
        
    Returns:
        Dictionary mapping element symbols to counts
        
    Raises:
        ValueError: If formula cannot be parsed
    """
    # Remove whitespace
    formula = formula.strip().replace(' ', '')
    
    if not formula:
        raise ValueError("Empty formula")
    
    # Handle parentheses recursively
    result = _parse_formula_recursive(formula)
    
    if not result:
        raise ValueError(f"Could not parse formula: {formula}")
    
    return result


def _parse_formula_recursive(formula: str) -> Dict[str, float]:
    """Recursively parse formula handling parentheses."""
    result = {}
    i = 0
    
    while i < len(formula):
        if formula[i] == '(':
            # Find matching closing parenthesis
            depth = 1
            j = i + 1
            while j < len(formula) and depth > 0:
                if formula[j] == '(':
                    depth += 1
                elif formula[j] == ')':
                    depth -= 1
                j += 1
            
            if depth != 0:
                raise ValueError(f"Unmatched parentheses in: {formula}")
            
            # Get content inside parentheses
            inner = formula[i+1:j-1]
            
            # Get multiplier after closing parenthesis
            k = j
            while k < len(formula) and (formula[k].isdigit() or formula[k] == '.'):
                k += 1
            
            multiplier = float(formula[j:k]) if j < k else 1.0
            
            # Recursively parse inner formula
            inner_result = _parse_formula_recursive(inner)
            
            # Apply multiplier and add to result
            for element, count in inner_result.items():
                result[element] = result.get(element, 0) + count * multiplier
            
            i = k
        else:
            # Try to match element
            match = ELEMENT_PATTERN.match(formula, i)
            if match:
                element = match.group(1)
                count_str = match.group(2)
                count = float(count_str) if count_str else 1.0
                
                result[element] = result.get(element, 0) + count
                i = match.end()
            else:
                raise ValueError(f"Unexpected character at position {i} in: {formula}")
    
    return result


def normalize_formula(formula: str) -> str:
    """
    Normalize a chemical formula for CrabNet.
    
    CrabNet requires formulas in the format: Element1<count1>Element2<count2>...
    where counts are explicit (no implicit 1s).
    
    Examples:
        AlLiSi -> Al1Li1Si1
        Fe2O3 -> Fe2O3
        H2O -> H2O1
        Ca(OH)2 -> Ca1H2O2
        
    Args:
        formula: Input chemical formula
        
    Returns:
        Normalized formula string
        
    Raises:
        ValueError: If formula cannot be parsed
    """
    # Parse the formula
    composition = parse_formula(formula)
    
    # Sort elements alphabetically for consistency
    sorted_elements = sorted(composition.keys())
    
    # Build normalized formula
    parts = []
    for element in sorted_elements:
        count = composition[element]
        # Format count: integer if whole number, else float
        if count == int(count):
            count_str = str(int(count))
        else:
            count_str = f"{count:.4f}".rstrip('0').rstrip('.')
        parts.append(f"{element}{count_str}")
    
    return ''.join(parts)


def validate_formula(formula: str) -> Tuple[bool, str]:
    """
    Validate a chemical formula.
    
    Args:
        formula: Chemical formula to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        composition = parse_formula(formula)
        
        # Check for valid elements
        for element in composition:
            if element not in ELEMENTS:
                return False, f"Unknown element: {element}"
        
        # Check for positive counts
        for element, count in composition.items():
            if count <= 0:
                return False, f"Invalid count for {element}: {count}"
        
        return True, "Valid formula"
    
    except ValueError as e:
        return False, str(e)


def get_composition_string(formula: str) -> str:
    """
    Get a human-readable composition string.
    
    Args:
        formula: Chemical formula
        
    Returns:
        Formatted composition string
    """
    composition = parse_formula(formula)
    parts = []
    for element, count in sorted(composition.items()):
        if count == int(count):
            parts.append(f"{element}: {int(count)}")
        else:
            parts.append(f"{element}: {count:.4f}")
    return ", ".join(parts)


# Test the module
if __name__ == "__main__":
    test_formulas = [
        "Fe2O3",
        "AlLiSi",
        "H2O",
        "Ca(OH)2",
        "NaCl",
        "Fe0.5Ni0.5O",
        "CaMg(CO3)2",
    ]
    
    print("Formula Normalization Tests:")
    print("-" * 50)
    
    for formula in test_formulas:
        try:
            normalized = normalize_formula(formula)
            print(f"{formula:20s} -> {normalized}")
        except ValueError as e:
            print(f"{formula:20s} -> ERROR: {e}")
