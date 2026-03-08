# PSMILESNet: CrabNet Adapted for Polymer SMILES

PSMILESNet is an adaptation of the CrabNet transformer architecture for predicting polymer properties from SMILES strings. This implementation focuses on predicting glass transition temperature (Tg) from polymer SMILES (PSMILES).

## Overview

CrabNet was originally designed for predicting inorganic material properties from chemical compositions. PSMILESNet adapts this architecture for organic polymers by:

1. **Atom-based featurization**: Instead of elemental composition, we use atom type counts and fractions from SMILES parsing
2. **Custom atom embeddings**: Chemical properties (electronegativity, atomic radius, valence) are used as initial atom embeddings
3. **SMILES parsing**: RDKit-based parsing with fallback for polymer-specific notation (connection points marked with `*`)

## Files

```
psmiles/
├── __init__.py              # Module initialization
├── smiles_featurizer.py     # SMILES parsing and featurization
├── psmiles_model.py         # PSMILESNet neural network architecture
├── train_psmiles.py         # Training wrapper and utilities
├── train_tg_psmiles.py      # Training script for Tg prediction
└── predict_tg.py            # Inference script for predictions
```

## Model Architecture

- **Atom Embedder**: Converts atom types to d-dimensional embeddings (default d=256)
- **Fractional Encoder**: Encodes atom fractions using positional encoding
- **Transformer Encoder**: 3 layers, 4 attention heads
- **Residual Network**: Feed-forward output network
- **Total Parameters**: ~4.5M

## Training Results

The model was trained on the polymer glass transition temperature dataset:

| Dataset    | MAE (°C) | RMSE (°C) | R²     |
|------------|----------|-----------|--------|
| Train      | 33.3     | 48.3      | 0.816  |
| Validation | 45.3     | 61.4      | 0.704  |
| Test       | 44.7     | 61.3      | 0.687  |

## Usage

### Making Predictions

```python
from psmiles.predict_tg import predict_tg, load_trained_model

# Predict Tg for a list of polymer SMILES
smiles_list = [
    '*CC(*)c1ccccc1',      # Polystyrene-like
    '*CC(*)(C)C(=O)OC',    # PMMA-like
    '*CC(*)C',             # Polypropylene-like
]

predictions, uncertainties = predict_tg(smiles_list)

for smiles, pred, unc in zip(smiles_list, predictions, uncertainties):
    print(f"{smiles}: Tg = {pred:.1f} ± {unc:.1f} °C")
```

### Command Line

```bash
# Predict from CSV file
python psmiles/predict_tg.py --input my_polymers.csv --output predictions.csv
```

The input CSV must have a `formula` column containing SMILES strings.

### Training a New Model

```python
from psmiles.train_tg_psmiles import main

# Train a new model on the Tg dataset
wrapper, results = main()
```

Or from command line:
```bash
python psmiles/train_tg_psmiles.py
```

## SMILES Format

The polymer SMILES format uses `*` to denote connection points in the polymer repeat unit:

- `*CC*` - Polyethylene repeat unit
- `*CC(*)c1ccccc1` - Polystyrene repeat unit
- `*CC(*)(C)C(=O)OC` - PMMA repeat unit

## Output Files

Training produces the following outputs:

- `models/trained_models/psmiles_tg.pth` - Trained model weights
- `figures/lc_data/psmiles_tg_lc.png` - Learning curve plot
- `psmiles_results/` - Evaluation results and plots:
  - `metrics.csv` - Performance metrics
  - `parity_*.png` - Predicted vs actual plots
  - `residuals_*.png` - Residual distribution plots
  - `predictions_*.csv` - Detailed predictions
  - `tg_distribution.png` - Dataset distribution
  - `metrics_summary.png` - Summary bar chart

## Dependencies

- PyTorch (CUDA recommended for GPU training)
- RDKit (for SMILES parsing)
- NumPy, Pandas, Matplotlib, Scikit-learn

## Notes

- The model uses a similar architecture to CrabNet but with modified embeddings
- Training completed in ~2 minutes on RTX 4060
- The uncertainty estimates are learned during training using robust L1 loss
- The `*` wildcard in SMILES is handled by mapping to a special token

## Citation

If you use this code, please cite the original CrabNet paper:

```
@article{wang2021crabnet,
  title={Compositionally restricted attention-based network for materials property predictions},
  author={Wang, Anthony Yu-Tung and Katsura, Yoichi and Deguchi, Kentaro and Ohta, Airi and Oono, Yoichi and Sugimoto, Toru and Takahashi, Tetsuya and Ozaki, Hiroaki and Yamamoto, Keisuke and Tanaka, Isao and Kimoto, Kazuhiko and Funahashi, Shunpei and Tanaka, Yutaro and Kondo, Yumiko and Amano, Keita and Nagao, Fumihiro and Ishi, Atsushi and Nakada, Mikiya and Li, Chao and Yuan, Zhao and Li, Zhesheng and Zhao, Jian-Kun and Li, Li and Soejima, Yoshitsugu and Kimura, Kosuke and Goto, Tsunehiro and Kuriyama, Nobuhiko and Nakamura, Shuichi and Nakamura, Atsuhiko and Kitamura, Tetsuya and Takagi, Tsuyoshi and Yamazaki, Taketo and Iiyama, Yasuaki and Inoue, Yoichi},
  journal={npj Computational Materials},
  volume={7},
  number={1},
  pages={77},
  year={2021},
  publisher={Nature Publishing Group}
}
```
