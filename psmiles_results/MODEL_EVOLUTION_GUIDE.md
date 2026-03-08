# Polymer Glass Transition Temperature (Tg) Prediction Models

## BioCrabNet: A Complete Beginner's Guide to Understanding the Three Model Versions

This document explains how we adapted CrabNet (a model for inorganic materials) to predict the glass transition temperature (Tg) of **polymers** from their chemical structure. This adaptation is called **BioCrabNet**. No machine learning knowledge required!

---

## Table of Contents

1. [What is CrabNet and Why Adapt It?](#what-is-crabnet-and-why-adapt-it)
2. [How BioCrabNet Relates to CrabNet](#how-biocrabnet-relates-to-crabnet)
3. [What Are We Trying to Do?](#what-are-we-trying-to-do)
4. [Key Concepts Explained Simply](#key-concepts-explained-simply)
5. [Version 1: PSMILESNet (Atom Counts)](#version-1-psmilesnet-atom-counts)
6. [Version 2: FingerprintNet (Single Model)](#version-2-fingerprintnet-single-model)
7. [Version 3: FingerprintNet Ensemble](#version-3-fingerprintnet-ensemble)
8. [Results Comparison](#results-comparison)
9. [Understanding the Graphs](#understanding-the-graphs)
10. [Understanding the Metrics](#understanding-the-metrics)
11. [Files and What They Contain](#files-and-what-they-contain)

---

## What is CrabNet and Why Adapt It?

### CrabNet: The Original Model

**CrabNet** (Compositionally Restricted Attention-Based Network) is a state-of-the-art deep learning model published in 2021 for predicting properties of **inorganic materials** from their chemical composition.

**What CrabNet does**:
- Input: Chemical formula like `Fe2O3` (iron oxide) or `LiCoO2` (lithium cobalt oxide)
- Output: Predicted property (e.g., band gap, formation energy, bulk modulus)

**Why CrabNet works well**:
1. Uses **element embeddings** (mat2vec) - learned vector representations of each element
2. Uses **transformer architecture** - the same technology behind ChatGPT
3. Uses **fractional encoding** - encodes how much of each element is present
4. Uses **attention mechanism** - learns which elements are most important for each property

### The Challenge: Polymers Are Different

CrabNet was designed for materials like `Fe2O3` where:
- Every atom of the same element behaves similarly
- The exact arrangement doesn't matter as much
- Composition (ratios) determines properties

**Polymers are different**:
- Same atoms can have wildly different arrangements
- A carbon in a ring vs. a carbon in a chain behaves differently
- **Structure matters**, not just composition!

### BioCrabNet: Our Adaptation

**BioCrabNet** adapts CrabNet's powerful architecture to work with **polymer SMILES** (text representation of molecular structure) instead of simple chemical formulas.

Think of it as:
- **CrabNet**: Designed for rocks and metals
- **BioCrabNet**: Designed for plastics and biological molecules

---

## How BioCrabNet Relates to CrabNet

### Architecture Comparison

Here's a side-by-side comparison of what we kept, modified, and added:

| Component | CrabNet (Original) | BioCrabNet Version 1 | BioCrabNet V2/V3 |
|-----------|-------------------|---------------------|------------------|
| **Input** | Chemical formula (`Fe2O3`) | Polymer SMILES (`*CC(*)c1ccccc1`) | Polymer SMILES |
| **Embeddings** | mat2vec (element embeddings) | Atom feature embeddings | Morgan fingerprints |
| **Positional Encoding** | Fractional encoder | Fractional encoder | StandardScaler normalization |
| **Main Architecture** | Transformer encoder | Transformer encoder | Residual network |
| **Attention Mechanism** | Yes (multi-head) | Yes (multi-head) | No (feedforward) |
| **Output Network** | Residual network | Residual network | Deep residual network |
| **Uncertainty** | Yes (learned) | Yes (learned) | Yes (learned) |

### Code Components Shared with CrabNet

**1. ResidualNetwork (Identical)**

The output layer that converts transformer output to predictions is **exactly the same** in both CrabNet and BioCrabNet:

```
CrabNet: kingcrab.py → class ResidualNetwork
BioCrabNet: psmiles_model.py → class ResidualNetwork (copied)
```

This network uses "skip connections" (residual connections) that help deep networks learn:
```
Input → Layer → ReLU → + → Output
   └──────────────────┘
        (skip/residual)
```

**2. FractionalEncoder (Identical)**

The encoding that represents "how much" of each element:

```
CrabNet: kingcrab.py → class FractionalEncoder
BioCrabNet: psmiles_model.py → class FractionalEncoder (copied)
```

Uses sinusoidal encoding inspired by the famous "Attention is All You Need" paper (Vaswani et al., 2017).

**3. Transformer Encoder Architecture**

```
CrabNet: 
    nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=2048)
    nn.TransformerEncoder(encoder_layer, num_layers=3)

BioCrabNet:
    nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=2048)
    nn.TransformerEncoder(encoder_layer, num_layers=3)
```

Same architecture, slightly smaller dimensions for our dataset.

**4. Attention Masking**

Both models use the same masking strategy to ignore padding:
```python
# Same in both CrabNet and BioCrabNet
mask = frac.unsqueeze(dim=-1)
mask = torch.matmul(mask, mask.transpose(-2, -1))
mask[mask != 0] = 1
src_mask = mask[:, 0] != 1
```

### What We Changed for Polymers

**1. Embedder: mat2vec → Atom Features**

CrabNet uses pre-trained element embeddings from materials science:
```python
# CrabNet: Uses mat2vec embeddings
mat2vec = 'data/element_properties/mat2vec.csv'
cbfv = pd.read_csv(mat2vec).values  # 200-dim vectors for 118 elements
```

BioCrabNet V1 creates custom atom features:
```python
# BioCrabNet: Custom atom features
atom_features = [atomic_number, period, group, electronegativity, 
                 atomic_radius, is_metal, ...]
```

**2. Input Processing: Composition → SMILES Parsing**

CrabNet takes element indices and fractions:
```python
# CrabNet input
src = [26, 8, 0, 0, ...]    # Element indices (Fe=26, O=8)
frac = [0.4, 0.6, 0, 0, ...]  # Atomic fractions
```

BioCrabNet V1 parses SMILES to get atom counts:
```python
# BioCrabNet V1 input (from SMILES "*CC(*)c1ccccc1")
src = [6, 1, 0, 0, ...]    # Atom types (C=6, H=1)
frac = [8, 8, 0, 0, ...]   # Atom counts (8 carbons, 8 hydrogens)
```

BioCrabNet V2/V3 uses molecular fingerprints:
```python
# BioCrabNet V2/V3 input
features = [0,1,0,1,0,0,1,..., 180.2, 5, 2, ...]  # 2048 fingerprint bits + 25 descriptors
```

### Technical Hyperparameters Comparison

| Parameter | CrabNet | BioCrabNet V1 | BioCrabNet V2/V3 |
|-----------|---------|---------------|------------------|
| `d_model` | 512 | 256 | N/A |
| `N` (layers) | 3 | 3 | 4 residual blocks |
| `heads` | 4 | 4 | N/A |
| `dim_feedforward` | 2048 | 2048 | 1024→512→256 |
| `dropout` | 0.1 | 0.1 | 0.05-0.15 |
| `batch_size` | 512 | 64 | 64-128 |
| `optimizer` | Lamb + Lookahead | AdamW | AdamW |

### Model Files Side-by-Side

```
CrabNet Structure:
├── crabnet/
│   ├── kingcrab.py      # CrabNet model (Embedder, Encoder, CrabNet)
│   └── model.py         # Training wrapper (Model class)
├── utils/
│   └── utils.py         # Data loaders, loss functions
└── data/
    └── element_properties/mat2vec.csv

BioCrabNet Structure:
├── psmiles/
│   ├── psmiles_model.py        # V1: PSMILESNet (adapted from kingcrab.py)
│   ├── smiles_featurizer.py    # SMILES parsing (replaces EDM_CsvLoader)
│   ├── fingerprint_model.py    # V2/V3: FingerprintNet
│   ├── fingerprint_featurizer.py  # Morgan fingerprints
│   ├── train_psmiles.py        # Training wrapper (like model.py)
│   └── train_tg_fingerprint.py # V2 training script
│   └── train_ensemble.py       # V3 ensemble training
└── data/
    └── polymer_smiles/glass_transistion_temperature.csv
```

### Why We Evolved Away from Pure CrabNet Architecture

**Version 1 Problem**: CrabNet's architecture assumes composition is key. But for polymers:

```
Same composition, different structures, different Tg:

Polyethylene:    *CC*       (Tg ≈ -125°C)
Polypropylene:   *CC(C)*    (Tg ≈ -10°C)  

Both are just C and H, but very different properties!
```

**Solution in V2/V3**: We kept CrabNet's residual network concept but switched to **fingerprints** that capture structure, not just composition.

### The CrabNet "DNA" in BioCrabNet

Even in V2/V3, we retain these CrabNet principles:

1. **Residual connections** - Skip connections in all dense layers
2. **Learned embeddings** - Features are transformed through learned layers
3. **Uncertainty quantification** - Model outputs confidence along with prediction
4. **Robust loss functions** - Same RobustL1/Huber loss for training stability
5. **Batch normalization** - Same regularization strategy
6. **PyTorch framework** - Same deep learning library

### BioCrabNet Name Explanation

**Bio** = Biological/organic molecules (polymers, SMILES, organic chemistry)
**Crab** = Based on CrabNet architecture and philosophy
**Net** = Neural network

The name reflects that this is CrabNet adapted for the "bio" world of organic molecules!

---

## What Are We Trying to Do?

### The Goal

We have a dataset of **7,174 polymers** (plastic-like materials). Each polymer has:
- A **SMILES string**: A text representation of the chemical structure (like `*CC(*)c1ccccc1`)
- A **glass transition temperature (Tg)**: The temperature at which the polymer goes from hard/glassy to soft/rubbery

**Our goal**: Given a new polymer's SMILES string, predict what its Tg will be.

### Why This Matters

- Measuring Tg experimentally takes time and money
- If we can predict it accurately from the chemical structure, we can quickly screen thousands of potential new polymers
- This helps in designing new materials for specific applications

### The Challenge

The Tg values in our dataset range from **-139°C to 495°C** (a span of 634°C). This is a huge range! Some polymers are flexible at room temperature, others are extremely rigid.

---

## Key Concepts Explained Simply

Before diving into the models, let's understand some terms:

### What is a "Model"?

Think of a model as a **formula or recipe that converts input into output**:
- **Input**: Chemical structure (SMILES string)
- **Output**: Predicted Tg temperature

The "training" process is like teaching the model by showing it thousands of examples: "Here's a polymer structure, and here's its actual Tg." The model gradually learns patterns.

### What is "Training"?

Imagine you're learning to estimate house prices. You look at many houses with known prices and start noticing patterns:
- Bigger houses cost more
- Houses in good neighborhoods cost more
- Newer houses cost more

After seeing enough examples, you can estimate prices for new houses. That's exactly what the model does with polymers and Tg!

### What is "MAE" (Mean Absolute Error)?

This tells us **how wrong the model's predictions are, on average**.

**Example**: If the model predicts these Tg values:
- Polymer A: Predicted 100°C, Actual 110°C → Error = 10°C
- Polymer B: Predicted 200°C, Actual 190°C → Error = 10°C  
- Polymer C: Predicted 50°C, Actual 80°C → Error = 30°C

**MAE = (10 + 10 + 30) / 3 = 16.7°C**

Lower MAE = Better predictions!

### What is "R²" (R-squared)?

This tells us **what percentage of the variation in Tg the model can explain**.

- **R² = 1.0 (100%)**: Perfect predictions
- **R² = 0.9 (90%)**: Excellent - model explains 90% of why some polymers have high/low Tg
- **R² = 0.7 (70%)**: Good - explains 70%
- **R² = 0.0 (0%)**: Useless - model is just guessing randomly

### What is a "Neural Network"?

A neural network is a type of model inspired by how brains work. It has layers of connected "neurons" that transform the input step by step until producing an output.

Think of it like a **factory assembly line**:
1. Raw materials (chemical structure) enter
2. Each station (layer) transforms the data
3. Final product (Tg prediction) comes out

---

## Version 1: PSMILESNet (Atom Counts)

**Location**: `psmiles_results/` (original results folder)

### CrabNet Similarity: ★★★★★ (Highest)

This version is the **most similar to CrabNet**. We essentially took CrabNet and swapped the input from inorganic formulas to polymer atom counts.

### How It Works

**The idea**: Count how many of each element are in the polymer, and use those counts to predict Tg. This mirrors CrabNet's approach of using element composition.

**Example for polystyrene** `*CC(*)c1ccccc1`:
- Carbon (C): 8 atoms
- Hydrogen (H): 8 atoms (implied)

The model receives: `[8, 8, 0, 0, ...]` (counts for C, H, N, O, etc.)

### Architecture (Network Structure) - Direct CrabNet Adaptation

```
Input: Atom counts (10 element types)
    ↓
Embedding: Convert each element count to a 256-dimensional vector
           [SAME AS CRABNET: Uses learned embeddings like mat2vec]
    ↓
Fractional Encoder: Encode atom fractions using sinusoidal functions
           [COPIED FROM CRABNET: FractionalEncoder class]
    ↓
Transformer Encoder: Process all elements together (3 layers, 4 heads)
           [SAME AS CRABNET: nn.TransformerEncoder]
    ↓
Attention Masking: Ignore padding elements
           [SAME AS CRABNET: src_key_padding_mask]
    ↓
Residual Network: Convert to single Tg prediction
           [COPIED FROM CRABNET: ResidualNetwork class]
    ↓
Output: Predicted Tg (°C) + Uncertainty
```

### Code Comparison

**CrabNet (kingcrab.py)**:
```python
class CrabNet(nn.Module):
    def __init__(self, out_dims=3, d_model=512, N=3, heads=4):
        self.encoder = Encoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
```

**BioCrabNet V1 (psmiles_model.py)**:
```python
class PSMILESNet(nn.Module):
    def __init__(self, out_dims=3, d_model=256, N=3, heads=4):
        self.encoder = SMILESEncoder(d_model=self.d_model, N=self.N, heads=self.heads)
        self.output_nn = ResidualNetwork(self.d_model, self.out_dims, self.out_hidden)
```

**The structures are nearly identical!** The main difference is `SMILESEncoder` uses atom type features instead of mat2vec element embeddings.

### The Problem With This Approach

**Atom counts lose structural information!**

Consider these two molecules:
- **Molecule A**: `CCCCCCCC` (straight chain of 8 carbons)
- **Molecule B**: `CC(C)(C)C(C)(C)C` (branched, same 8 carbons)

Both have the same atom counts (C=8, H=18), but they have very different properties because their **structures** are different!

This is like describing a house only by counting materials:
- "This house has 10,000 bricks, 500 wooden beams, 50 windows"
- That tells you nothing about the floor plan, number of rooms, or layout!

### Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 31.52°C | 43.01°C | 44.73°C |
| RMSE | 49.17°C | 58.98°C | 61.30°C |
| R² | 0.810 | 0.711 | 0.687 |

**Interpretation**:
- On average, predictions are **44.73°C away** from the actual temperature
- The model explains only **68.7%** of why some polymers have different Tg values
- There's a big gap between Train (31.52°C) and Test (44.73°C), suggesting **overfitting** (the model memorized training examples instead of learning general patterns)

---

## Version 2: FingerprintNet (Single Model)

**Location**: `psmiles_results_v2/`

### CrabNet Similarity: ★★★☆☆ (Medium)

This version keeps CrabNet's **residual network** philosophy but replaces the transformer with a deep feedforward network. We discovered that for polymers, structural fingerprints work better than CrabNet's composition-based approach.

### What Changed

Instead of just counting atoms, we now capture **structural information**!

### What is a "Morgan Fingerprint"?

A fingerprint is a way to encode a molecule's structure as a long list of 0s and 1s.

**How it works** (simplified):
1. Look at each atom in the molecule
2. Look at what's connected to it (neighbors, rings, branches)
3. Create a unique "code" for that local environment
4. Repeat for all atoms
5. Combine into a 2048-bit binary "fingerprint"

**Example**:
- Bit #42 might mean "there's a benzene ring (aromatic 6-carbon ring)"
- Bit #567 might mean "there's a carbonyl group (C=O)"
- Bit #1234 might mean "there's a branching point"

This captures structure, not just composition!

### Molecular Descriptors (25 additional features)

We also calculate 25 properties of the molecule:
- **MolWt**: Molecular weight (heavier = higher Tg generally)
- **NumRotatableBonds**: How flexible is the molecule backbone?
- **NumAromaticRings**: How many ring structures?
- **TPSA**: Total polar surface area
- **FractionCSP3**: What fraction of carbons are sp3 hybridized (tetrahedral)?
- And 20 more...

### Total Features

- **2048** fingerprint bits (structure)
- **25** molecular descriptors (properties)
- **= 2073** total features per polymer

Compare to Version 1's **10** features (atom counts)!

### Architecture - CrabNet Components Retained

```
Input: 2073 features (fingerprint + descriptors)
    ↓
Normalize: Scale all features to similar ranges
    ↓
Dense Layer: 2073 → 1024 neurons
    ↓
Residual Blocks (4x): Process and refine
    [FROM CRABNET: ResidualNetwork with skip connections]
    ↓
Dense Layer: → 512 neurons
    [FROM CRABNET: Same LeakyReLU activation]
    ↓
Dense Layer: → 256 neurons
    ↓
Output: Predicted Tg + Uncertainty
    [FROM CRABNET: Same uncertainty quantification approach]
```

**CrabNet components retained**:
- ✓ Residual connections (skip connections)
- ✓ LeakyReLU activations
- ✓ Uncertainty estimation in output
- ✓ Huber loss / RobustL1 loss function
- ✓ AdamW-like optimizer (CrabNet uses Lamb + Lookahead)
- ✗ Transformer encoder (replaced with feedforward)

### What is a "Residual Block"?

A residual block is a clever trick: instead of just transforming data, **it also adds the original input back**:

```
Input → Transform → + Input → Output
          ↑___________↓ (shortcut connection)
```

This helps the network learn better because:
- Information can flow directly through shortcuts
- The network only needs to learn the "difference" (residual)
- Prevents the "vanishing gradient" problem (where learning stops in deep networks)

### Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| MAE | 9.57°C | 26.68°C | **25.51°C** |
| RMSE | 15.91°C | 40.20°C | 41.46°C |
| R² | 0.980 | 0.873 | **0.857** |

**Interpretation**:
- Test MAE improved from 44.73°C to **25.51°C** (43% better!)
- R² improved from 0.687 to **0.857** (explains 86% of variation now)
- Still some overfitting (Train MAE much lower than Test MAE)

### Why Is It Better?

The fingerprint captures things like:
- "This polymer has benzene rings" → usually higher Tg (stiff rings)
- "This polymer has many rotatable bonds" → usually lower Tg (flexible)
- "This polymer has branching" → affects packing and Tg

These structural features are what actually determine Tg!

---

## Version 3: FingerprintNet Ensemble

**Location**: `psmiles_results_v3/`

### CrabNet Similarity: ★★★☆☆ (Medium)

Same architecture as V2, but uses CrabNet-inspired **ensemble training** strategy (CrabNet also uses 5-fold cross-validation for benchmarking).

### What is an Ensemble?

Instead of training **one** model, we train **five** models and average their predictions!

**Analogy**: Imagine you want to estimate someone's age from their photo.
- Person A guesses: 35
- Person B guesses: 32
- Person C guesses: 38
- Person D guesses: 34
- Person E guesses: 36

**Average: 35** - This is probably more accurate than any single guess!

Each model might make different mistakes, but averaging cancels out individual errors.

### What is Cross-Validation?

Instead of using the same train/test split, we rotate which data is used for training:

```
Fold 1: [Test][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train]
Fold 3: [Train][Train][Test][Train][Train]
Fold 4: [Train][Train][Train][Test][Train]
Fold 5: [Train][Train][Train][Train][Test]
```

Each fold produces one trained model. We keep all 5 models for the ensemble.

**Benefits**:
- Uses all data for both training and validation
- More robust estimate of true performance
- 5 diverse models that make different mistakes

### Hyperparameter Search

Before final training, we tested 6 different configurations:
- Different network sizes (wider/narrower)
- Different numbers of residual blocks
- Different dropout rates (regularization)

The best configuration was:
- Hidden layers: [1024, 512, 256]
- Residual blocks: 4
- Dropout: 0.05
- Learning rate: 2×10⁻⁴

### Results

| Metric | CV Mean | Test (Ensemble) |
|--------|---------|-----------------|
| MAE | 25.61°C | **23.58°C** |
| RMSE | - | 38.01°C |
| R² | - | **0.880** |

**Fold-by-Fold CV Results**:
| Fold | Validation MAE |
|------|----------------|
| 1 | 26.56°C |
| 2 | 26.93°C |
| 3 | 25.24°C |
| 4 | 24.28°C |
| 5 | 25.04°C |
| **Mean** | **25.61°C** (±0.99) |

**Interpretation**:
- Test MAE improved to **23.58°C** (best result!)
- R² improved to **0.880** (88% of variation explained)
- Consistent performance across folds (small ±0.99 variation)
- Ensemble averaging reduced test error by 1.93°C vs single model

---

## Results Comparison

### Summary Table

| Version | Model Type | Test MAE | Test R² | Improvement |
|---------|-----------|----------|---------|-------------|
| V1 | PSMILESNet (atom counts) | 44.73°C | 0.687 | Baseline |
| V2 | FingerprintNet (single) | 25.51°C | 0.857 | **-43%** MAE |
| V3 | FingerprintNet (ensemble) | **23.58°C** | **0.880** | **-47%** MAE |

### What Each Version Added

| Version | Key Innovation | Why It Helped |
|---------|----------------|---------------|
| V1 → V2 | Morgan fingerprints + descriptors | Captured molecular structure, not just composition |
| V2 → V3 | Ensemble of 5 models + hyperparameter search | Reduced variance, found optimal settings |

### Visual Progress

```
MAE (lower is better):
V1: ████████████████████████████████████████████ 44.73°C
V2: █████████████████████████ 25.51°C
V3: ███████████████████████ 23.58°C

R² (higher is better, max 1.0):
V1: ████████████████████████████████████████████████████████████████████ 0.687
V2: █████████████████████████████████████████████████████████████████████████████████████ 0.857  
V3: ████████████████████████████████████████████████████████████████████████████████████████ 0.880
```

---

## Understanding the Graphs

### 1. Parity Plot (Actual vs Predicted)

**What it shows**: Each point is one polymer. X-axis is actual Tg, Y-axis is predicted Tg.

**How to read it**:
- The diagonal dashed line is "perfect prediction" (predicted = actual)
- Points close to the line = good predictions
- Points far from the line = errors
- Color indicates uncertainty (how confident the model is)

**Good parity plot**:
```
    Predicted
         │     .**
         │   .**
         │ .**       Points cluster
         │**         tightly around
         ├──────────  the diagonal
               Actual
```

**Bad parity plot**:
```
    Predicted
         │  * *    *
         │    *  *     Points scattered
         │ *   *       everywhere
         │*   *    *
         ├──────────
               Actual
```

### 2. Residuals Histogram

**What it shows**: Distribution of errors (Predicted - Actual).

**How to read it**:
- Center (0) means no error
- Bars should be centered around 0 (no systematic bias)
- Narrow distribution = small errors
- Wide distribution = large errors

**Good residuals**:
```
Count │     ▄▄▄
      │   ▄█████▄
      │ ▄█████████▄     Narrow, centered at 0
      ├───────────────
        -20   0   +20
              Error (°C)
```

**Bad residuals**:
```
Count │▄       ▄
      │██     ██         Wide, possibly biased
      │███▄▄▄███
      ├───────────────
        -100  0   +100
              Error (°C)
```

### 3. Learning Curve

**What it shows**: How the model improved during training.

**How to read it**:
- X-axis: Epoch (one pass through all training data)
- Y-axis: MAE (error)
- Two lines: Training error and Validation error

**Ideal learning curve**:
```
MAE │\
    │ \  ____         Both lines decrease
    │  \/    \____    and converge together
    │   training  ──────
    │   validation ─ ─ ─
    ├───────────────────
           Epochs
```

**Overfitting (bad)**:
```
MAE │\
    │ \               Training keeps decreasing
    │  \____          but validation goes up
    │       \______ training
    │   ─ ─ ─ ─ ─ ─ validation
    ├───────────────────
           Epochs
```

### 4. Cross-Validation Bar Chart (V3 only)

**What it shows**: How each of the 5 folds performed.

**How to read it**:
- Each bar is one fold's validation MAE
- Red dashed line is the average
- Similar heights = consistent model
- Wildly different heights = unstable model

**Good CV results**:
```
MAE │
 30 │ ▓▓  ▓▓  ▓▓  ▓▓  ▓▓    Similar heights
 25 │ ██──██──██──██──██─── Average line
 20 │
    ├───────────────────
      F1  F2  F3  F4  F5
```

---

## Understanding the Metrics

### MAE (Mean Absolute Error)

**Formula**: Average of |Predicted - Actual| across all samples

**Our Results**:
- V1: 44.73°C → "On average, we're off by 45°C"
- V2: 25.51°C → "On average, we're off by 26°C"
- V3: 23.58°C → "On average, we're off by 24°C"

**Context**: The Tg range is 634°C (-139 to 495), so:
- V1 error is 7.1% of the range
- V3 error is 3.7% of the range

### RMSE (Root Mean Squared Error)

**Formula**: Square root of average of (Predicted - Actual)² 

**Why it's higher than MAE**: Squaring penalizes large errors more heavily.

If MAE is 25°C but RMSE is 40°C, it means there are some predictions with large errors (outliers).

### R² (Coefficient of Determination)

**Interpretation**:
- V1 R² = 0.687: "The model explains 68.7% of why Tg varies between polymers"
- V3 R² = 0.880: "The model explains 88.0% of why Tg varies between polymers"

**What's the other 12%?** Things the model doesn't capture:
- Measurement error in the original data
- Very subtle structural features
- Molecular weight distribution
- Processing conditions

---

## Files and What They Contain

### psmiles_results/ (Version 1)

| File | Contents |
|------|----------|
| `metrics.csv` | MAE, RMSE, R² for train/val/test |
| `predictions_train.csv` | All training set predictions |
| `predictions_validation.csv` | All validation set predictions |
| `predictions_test.csv` | All test set predictions |

### psmiles_results_v2/ (Version 2)

| File | Contents |
|------|----------|
| `metrics.csv` | MAE, RMSE, R² for train/val/test |
| `predictions_train.csv` | Training predictions with uncertainties |
| `predictions_val.csv` | Validation predictions with uncertainties |
| `predictions_test.csv` | Test predictions with uncertainties |
| `learning_curve.png` | Training progress over epochs |
| `parity_train.png` | Parity plot for training set |
| `parity_val.png` | Parity plot for validation set |
| `parity_test.png` | Parity plot for test set |

### psmiles_results_v3/ (Version 3)

| File | Contents |
|------|----------|
| `predictions_test.csv` | Test predictions with ensemble uncertainty |
| `ensemble_results.png` | Combined plot with parity, residuals, CV scores |

### Prediction CSV Columns

| Column | Meaning |
|--------|---------|
| `formula` | SMILES string of the polymer |
| `actual` | True measured Tg (°C) |
| `predicted` | Model's predicted Tg (°C) |
| `uncertainty` | How confident the model is (lower = more confident) |
| `residual` | Error: predicted - actual (°C) |

---

## Practical Interpretation

### How Good Is 23.58°C MAE?

**Context**:
- Experimental Tg measurement error is typically ±2-10°C
- Tg depends on measurement method (DSC vs DMA)
- Different literature sources report different values for the same polymer

**For practical use**:
- **Great for screening**: If you need to identify polymers with Tg > 200°C, this model is reliable
- **Good for ranking**: Comparing which polymer has higher/lower Tg works well
- **Decent for estimation**: ±24°C gives a reasonable ballpark

**What 23.58°C means in practice**:
- If model says Tg = 100°C, actual is likely between 76-124°C
- If model says Tg = 300°C vs Tg = 100°C, you can trust that difference

### When to Use Which Model?

| Use Case | Recommended Model |
|----------|-------------------|
| Quick screening | V2 (Single FingerprintNet) - Fast |
| Best accuracy | V3 (Ensemble) - Most accurate |
| Understanding model | V1 - Simpler, easier to interpret |

---

## Key Takeaways

1. **Better input = Better output**: Fingerprints capture structure; atom counts don't
2. **Ensemble averaging works**: Combining 5 models beats any single model
3. **The final model achieves**:
   - 23.58°C average error (3.7% of the Tg range)
   - 88% of Tg variation explained
   - 47% improvement over the baseline

4. **Limitations**:
   - Some polymers are harder to predict than others
   - The model works best within the range of training data
   - Very unusual structures may have higher errors

---

## Technical Details (For Reference)

### Software Used
- Python 3.10
- PyTorch 2.6.0 (CUDA 12.4)
- RDKit 2025.9.6
- scikit-learn 1.x

### Hardware
- GPU: NVIDIA RTX 4060 Laptop (8GB)
- Training time: V3 ensemble took ~10 minutes

### Model Files
- `models/trained_models/psmiles_tg.pth` - Version 1
- `models/trained_models/psmiles_tg_fingerprint.pth` - Version 2
- `models/trained_models/psmiles_tg_ensemble.pth` - Version 3 (recommended)

---

## Summary: BioCrabNet's Relationship to CrabNet

### What We Inherited from CrabNet

| Component | CrabNet Original | BioCrabNet | Purpose |
|-----------|-----------------|------------|---------|
| **ResidualNetwork** | ✓ | ✓ (identical code) | Output prediction layer with skip connections |
| **FractionalEncoder** | ✓ | ✓ (V1 only) | Encode element amounts using sinusoidal functions |
| **TransformerEncoder** | ✓ | ✓ (V1 only) | Self-attention to learn element interactions |
| **Uncertainty Output** | ✓ | ✓ | Model predicts confidence along with value |
| **RobustL1/Huber Loss** | ✓ | ✓ | Training loss that handles outliers |
| **5-Fold CV** | ✓ | ✓ (V3) | Cross-validation for robust evaluation |
| **Element Embeddings** | mat2vec | Custom atom features | Represent atoms as vectors |

### Why BioCrabNet Evolved

```
CrabNet (Inorganic Materials)
    │
    │  Problem: Polymers need structural info, not just composition
    │
    ├──> V1: PSMILESNet (Direct CrabNet adaptation)
    │         └─ Kept transformer, changed embeddings
    │         └─ Result: 44.73°C MAE (not good enough)
    │
    ├──> V2: FingerprintNet (Kept ResidualNetwork, new features)
    │         └─ Morgan fingerprints capture structure
    │         └─ Result: 25.51°C MAE (43% improvement!)
    │
    └──> V3: Ensemble (CrabNet-style cross-validation)
              └─ 5-fold CV ensemble like CrabNet benchmarks
              └─ Result: 23.58°C MAE (best!)
```

### The BioCrabNet Philosophy

**Same as CrabNet**:
1. Learn from chemical structure → predict property
2. Use deep neural networks with residual connections
3. Quantify uncertainty in predictions
4. Validate rigorously with cross-validation

**Adapted for polymers**:
1. Replace composition with structure (fingerprints)
2. Add molecular descriptors for global properties
3. Ensemble models for biological data variability

### If You're Presenting This Project

You can say:

> "BioCrabNet adapts the CrabNet architecture for polymer property prediction. While CrabNet uses element composition and transformer attention for inorganic materials, BioCrabNet uses molecular fingerprints and residual networks for organic polymers. The core philosophy remains the same: learn chemical structure-property relationships using deep learning. Key CrabNet components like the ResidualNetwork, uncertainty quantification, and cross-validation strategy are preserved. Our adaptation achieves a 47% improvement over the baseline, demonstrating that CrabNet's principles can be successfully extended to biological/organic materials."

---

## Technical Appendix: Code Lineage

### Files Copied/Adapted from CrabNet

| BioCrabNet File | CrabNet Source | Changes Made |
|-----------------|----------------|--------------|
| `psmiles/psmiles_model.py` | `crabnet/kingcrab.py` | Changed Embedder to SMILESEmbedder |
| `psmiles/train_psmiles.py` | `crabnet/model.py` | Changed data loader for SMILES |
| `psmiles/fingerprint_model.py` | `crabnet/kingcrab.py` | Kept ResidualNetwork concept |

### Shared Code Example

**ResidualNetwork (identical in both)**:
```python
class ResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)  # Skip connection!
        return self.fc_out(fea)
```

This exact class appears in:
- `crabnet/kingcrab.py` (line 15-44)
- `psmiles/psmiles_model.py` (line 24-42)

**The lineage is clear and documented!**
