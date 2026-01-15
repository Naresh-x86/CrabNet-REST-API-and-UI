# CrabNet Material Property Prediction API - Complete Frontend Integration Guide

## Overview

This document provides comprehensive documentation for the CrabNet Material Property Prediction API. This API enables:

1. **Material Property Prediction** - Predict 83+ material properties from chemical formulas using CrabNet neural network
2. **Materials Project Integration** - Search, retrieve, and explore material data from the Materials Project database
3. **Structure Visualization** - Access crystal structure information and images
4. **Natural Language Descriptions** - Generate human-readable descriptions of crystal structures

**Base URL:** `http://localhost:8000`

**Interactive Documentation:** 
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Table of Contents

1. [API Endpoints Summary](#api-endpoints-summary)
2. [Endpoint Details](#endpoint-details)
   - [GET /get-models](#get-get-models)
   - [GET /autocomplete-search](#get-autocomplete-search)
   - [GET /retrieve](#get-retrieve)
   - [GET /summary](#get-summary)
   - [GET /related-materials](#get-related-materials)
   - [GET /natural-language-summary](#get-natural-language-summary)
   - [GET /predict](#get-predict)
   - [GET /health](#get-health)
3. [Data Models](#data-models)
4. [Property Units Reference](#property-units-reference)
5. [Error Handling](#error-handling)
6. [Frontend Implementation Guide](#frontend-implementation-guide)
7. [Example User Flows](#example-user-flows)

---

## API Endpoints Summary

| Endpoint | Method | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `/get-models` | GET | List available prediction models | None |
| `/autocomplete-search` | GET | Autocomplete formula search | `formula` |
| `/retrieve` | GET | Search materials by formula | `formula` |
| `/summary` | GET | Get full material details | `material_id` |
| `/related-materials` | GET | Find similar materials | `material_id` |
| `/natural-language-summary` | GET | Generate text description | `material_id` |
| `/predict` | GET | Predict material property | `formula`, `property_name` |
| `/health` | GET | API health check | None |

---

## Endpoint Details

### GET /get-models

**Purpose:** Retrieve a list of all available CrabNet models for property prediction. This should be called first to populate a dropdown/selection for the user to choose which property they want to predict.

**URL:** `/get-models`

**Parameters:** None

**Response Format:**
```typescript
Array<{
  name: string;         // Model identifier (use this for /predict)
  description: string;  // Human-readable description
  units: string;        // Units of the predicted value
}>
```

**Example Response:**
```json
[
  {
    "name": "OQMD_Bandgap",
    "description": "Band gap energy from OQMD database",
    "units": "eV"
  },
  {
    "name": "aflow__ael_bulk_modulus_vrh",
    "description": "Bulk modulus (VRH average)",
    "units": "GPa"
  },
  {
    "name": "OQMD_Formation_Enthalpy",
    "description": "Formation enthalpy per atom",
    "units": "eV/atom"
  },
  {
    "name": "aflow__agl_thermal_conductivity_300K",
    "description": "Thermal conductivity at 300K",
    "units": "W/(m·K)"
  }
]
```

**Frontend Usage:**
- Call this on app initialization
- Populate a dropdown/select component with model names
- Display description as tooltip or helper text
- Store units to display with prediction results

**Example Fetch:**
```javascript
const response = await fetch('http://localhost:8000/get-models');
const models = await response.json();
// models is an array of {name, description, units}
```

---

### GET /autocomplete-search

**Purpose:** Provide autocomplete suggestions as the user types a chemical formula. This helps users discover valid formulas and prevents typos.

**URL:** `/autocomplete-search`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `formula` | string | Yes | Partial formula to search (e.g., "Fe2O", "Al", "Ti") |

**Response Format:**
```typescript
Array<{
  formula_pretty: string;  // Complete formula suggestion
}>
```

**Example Request:**
```
GET /autocomplete-search?formula=Fe2O
```

**Example Response:**
```json
[
  {"formula_pretty": "Fe2OF2"},
  {"formula_pretty": "Fe2OF3"},
  {"formula_pretty": "Rb4Fe2O"},
  {"formula_pretty": "Zr6Fe2O"},
  {"formula_pretty": "Ti4Fe2O"},
  {"formula_pretty": "Fe2(CO)9"},
  {"formula_pretty": "SrFe2S2O"},
  {"formula_pretty": "Li3Fe2OF5"},
  {"formula_pretty": "LiFe2(ClO)2"},
  {"formula_pretty": "Sr3Fe2(HO)12"}
]
```

**Frontend Usage:**
- Debounce input (300-500ms recommended)
- Call on each keystroke after debounce
- Display suggestions in a dropdown below input
- Allow user to select a suggestion or type their own
- Minimum 1-2 characters before searching

**Example Fetch:**
```javascript
const response = await fetch(`http://localhost:8000/autocomplete-search?formula=${encodeURIComponent(userInput)}`);
const suggestions = await response.json();
// suggestions is an array of {formula_pretty}
```

**Notes:**
- Returns up to 10 results (hardcoded limit)
- Results come from Materials Project database
- Empty array returned if no matches

---

### GET /retrieve

**Purpose:** Search for materials matching a specific formula. Returns a list of materials with basic information. Each result has a unique `material_id` that can be used to get more details.

**URL:** `/retrieve`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `formula` | string | Yes | Complete chemical formula (e.g., "Fe2O3", "TiO2") |

**Response Format:**
```typescript
{
  data: Array<{
    material_id: string;      // Unique ID (e.g., "mp-19770")
    formula_pretty: string;   // Formatted formula
    volume: number;           // Unit cell volume (Å³)
    density: number;          // Density (g/cm³)
    symmetry: {
      crystal_system: string; // e.g., "Trigonal", "Cubic"
      symbol: string;         // Space group symbol
      number: number;         // Space group number
      point_group: string;    // Point group symbol
      symprec: number;
      angle_tolerance: number;
      version: string;
    }
  }>;
  meta: {
    api_version: string;
    time_stamp: string;
    total_doc: number;        // Total matching documents
    max_limit: number;
    default_fields: string[];
  }
}
```

**Example Request:**
```
GET /retrieve?formula=Fe2O3
```

**Example Response:**
```json
{
  "data": [
    {
      "formula_pretty": "Fe2O3",
      "volume": 103.1107530196113,
      "density": 5.143372282900504,
      "symmetry": {
        "crystal_system": "Trigonal",
        "symbol": "R-3c",
        "number": 167,
        "point_group": "-3m",
        "symprec": 0.1,
        "angle_tolerance": 5,
        "version": "2.5.0"
      },
      "material_id": "mp-19770"
    },
    {
      "formula_pretty": "Fe2O3",
      "volume": 982.2523359429204,
      "density": 4.319354363385391,
      "symmetry": {
        "crystal_system": "Triclinic",
        "symbol": "P1",
        "number": 1,
        "point_group": "1",
        "symprec": 0.1,
        "angle_tolerance": 5,
        "version": "2.5.0"
      },
      "material_id": "mp-1244869"
    }
  ],
  "meta": {
    "api_version": "0.86.3.dev4+g74194bbf8",
    "time_stamp": "2026-01-15T15:30:33.870915+00:00",
    "total_doc": 26
  }
}
```

**Frontend Usage:**
- Display results in a table or card list
- Show material_id, crystal_system, density, volume
- Make each result clickable to view full details
- Use material_id for subsequent API calls

**Example Fetch:**
```javascript
const response = await fetch(`http://localhost:8000/retrieve?formula=${encodeURIComponent(formula)}`);
const result = await response.json();
// result.data is array of materials
// result.meta.total_doc shows total matches
```

**Notes:**
- Returns up to 10 results per request
- Multiple polymorphs/structures may exist for same formula
- material_id format is "mp-XXXXX"

---

### GET /summary

**Purpose:** Get comprehensive information about a specific material. This returns all available data from Materials Project including electronic, magnetic, and structural properties.

**URL:** `/summary`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `material_id` | string | Yes | Material ID (e.g., "mp-19770") |

**Response Format:**
```typescript
{
  data: Array<{
    // Basic Info
    material_id: string;
    formula_pretty: string;
    formula_anonymous: string;
    chemsys: string;              // Element system (e.g., "Fe-O")
    nelements: number;
    elements: string[];
    nsites: number;
    volume: number;
    density: number;
    density_atomic: number;
    
    // Symmetry
    symmetry: {
      crystal_system: string;
      symbol: string;
      number: number;
      point_group: string;
    };
    
    // Thermodynamic
    formation_energy_per_atom: number;
    energy_per_atom: number;
    energy_above_hull: number;
    is_stable: boolean;
    equilibrium_reaction_energy_per_atom: number;
    
    // Electronic
    band_gap: number;
    cbm: number | null;           // Conduction band minimum
    vbm: number | null;           // Valence band maximum
    efermi: number;               // Fermi energy
    is_gap_direct: boolean;
    is_metal: boolean;
    
    // Magnetic
    is_magnetic: boolean;
    ordering: string;             // e.g., "FM", "AFM", "NM"
    total_magnetization: number;
    total_magnetization_normalized_vol: number;
    total_magnetization_normalized_formula_units: number;
    
    // Dielectric
    e_total: number;
    e_ionic: number;
    e_electronic: number;
    n: number;                    // Refractive index
    
    // Elastic (may be null)
    bulk_modulus: number | null;
    shear_modulus: number | null;
    universal_anisotropy: number | null;
    homogeneous_poisson: number | null;
    
    // Structure
    structure: {
      lattice: {
        a: number;
        b: number;
        c: number;
        alpha: number;
        beta: number;
        gamma: number;
        volume: number;
        matrix: number[][];
      };
      sites: Array<{
        species: Array<{element: string, occu: number}>;
        abc: number[];            // Fractional coordinates
        xyz: number[];            // Cartesian coordinates
        label: string;
      }>;
    };
    
    // Metadata
    last_updated: string;
    deprecated: boolean;
    theoretical: boolean;
    possible_species: string[];   // e.g., ["Fe3+", "O2-"]
    
    // Available data flags
    has_props: {
      materials: boolean;
      thermo: boolean;
      xas: boolean;
      grain_boundaries: boolean;
      electronic_structure: boolean;
      magnetism: boolean;
      elasticity: boolean;
      dielectric: boolean;
      piezoelectric: boolean;
      surface_properties: boolean;
      phonon: boolean;
      eos: boolean;
      dos: boolean;
      bandstructure: boolean;
      provenance: boolean;
      charge_density: boolean;
      chemenv: boolean;
      substrates: boolean;
    };
    
    // External database IDs
    database_IDs: {
      icsd: number[];
      cod: number[];
      // ... other databases
    };
  }>;
  meta: {
    api_version: string;
    time_stamp: string;
    total_doc: number;
  }
}
```

**Example Request:**
```
GET /summary?material_id=mp-19770
```

**Example Response (truncated for brevity):**
```json
{
  "data": [
    {
      "material_id": "mp-19770",
      "formula_pretty": "Fe2O3",
      "nelements": 2,
      "elements": ["Fe", "O"],
      "nsites": 10,
      "volume": 103.1107530196113,
      "density": 5.143372282900504,
      "symmetry": {
        "crystal_system": "Trigonal",
        "symbol": "R-3c",
        "number": 167,
        "point_group": "-3m"
      },
      "formation_energy_per_atom": -1.7070913529999991,
      "energy_above_hull": 0,
      "is_stable": true,
      "band_gap": 0,
      "is_metal": true,
      "is_magnetic": true,
      "ordering": "FM",
      "total_magnetization": 19.9980745,
      "e_total": 19.869691523879137,
      "n": 2.492665129270376,
      "structure": {
        "lattice": {
          "a": 5.0702,
          "b": 5.0702,
          "c": 13.8939,
          "alpha": 90,
          "beta": 90,
          "gamma": 120,
          "volume": 103.11075301961128
        },
        "sites": [
          {
            "species": [{"element": "Fe", "occu": 1}],
            "abc": [0.0, 0.0, 0.35528],
            "xyz": [0.0, 0.0, 4.9358],
            "label": "Fe"
          }
        ]
      },
      "has_props": {
        "materials": true,
        "thermo": true,
        "electronic_structure": true,
        "magnetism": true,
        "dielectric": true
      },
      "possible_species": ["Fe3+", "O2-"]
    }
  ],
  "meta": {
    "total_doc": 1
  }
}
```

**Frontend Usage:**
- Display comprehensive material information
- Show properties in organized sections (basic, electronic, magnetic, etc.)
- Use `structure.sites` for 3D visualization
- Use `has_props` to show/hide sections based on available data
- Format numbers appropriately (e.g., 2-4 decimal places)

**Key Properties to Display:**

| Property | Display Name | Format |
|----------|-------------|--------|
| `formula_pretty` | Formula | As-is |
| `symmetry.crystal_system` | Crystal System | As-is |
| `symmetry.symbol` | Space Group | As-is |
| `density` | Density | X.XXX g/cm³ |
| `volume` | Volume | X.XX ų |
| `band_gap` | Band Gap | X.XXX eV |
| `is_metal` | Metallic | Yes/No |
| `is_magnetic` | Magnetic | Yes/No |
| `formation_energy_per_atom` | Formation Energy | X.XXXX eV/atom |
| `energy_above_hull` | E Above Hull | X.XXXX eV/atom |
| `is_stable` | Thermodynamically Stable | Yes/No |

**Example Fetch:**
```javascript
const response = await fetch(`http://localhost:8000/summary?material_id=${materialId}`);
const result = await response.json();
const material = result.data[0]; // Single material object
```

---

### GET /related-materials

**Purpose:** Find materials that are structurally similar to a given material. Returns top 5 similar materials with similarity scores and structure image URLs.

**URL:** `/related-materials`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `material_id` | string | Yes | Material ID (e.g., "mp-19770") |

**Response Format:**
```typescript
Array<{
  material_id: string;    // MP ID of similar material
  formula: string;        // Chemical formula
  similarity: number;     // Similarity score (0-100, higher = more similar)
  image_url: string;      // URL to structure image PNG
}>
```

**Example Request:**
```
GET /related-materials?material_id=mp-19770
```

**Example Response:**
```json
[
  {
    "material_id": "mp-1243",
    "formula": "Ga4O6",
    "similarity": 96.24,
    "image_url": "https://materialsproject-build.s3.amazonaws.com/images/structures/mp-1243.png"
  },
  {
    "material_id": "mp-22375",
    "formula": "In4S6",
    "similarity": 91.08,
    "image_url": "https://materialsproject-build.s3.amazonaws.com/images/structures/mp-22375.png"
  },
  {
    "material_id": "mp-755313",
    "formula": "Sc4O6",
    "similarity": 90.5,
    "image_url": "https://materialsproject-build.s3.amazonaws.com/images/structures/mp-755313.png"
  },
  {
    "material_id": "mp-1143",
    "formula": "Al4O6",
    "similarity": 90.32,
    "image_url": "https://materialsproject-build.s3.amazonaws.com/images/structures/mp-1143.png"
  },
  {
    "material_id": "mp-1047",
    "formula": "Ca6N4",
    "similarity": 89.29,
    "image_url": "https://materialsproject-build.s3.amazonaws.com/images/structures/mp-1047.png"
  }
]
```

**Frontend Usage:**
- Display as a grid of cards with structure images
- Show similarity as percentage or progress bar
- Make cards clickable to view that material's details
- Images are PNG format, typically 300x300px

**Example Fetch:**
```javascript
const response = await fetch(`http://localhost:8000/related-materials?material_id=${materialId}`);
const similarMaterials = await response.json();
// Array of {material_id, formula, similarity, image_url}
```

**Notes:**
- Returns exactly 5 results
- Similarity calculated as `100 - dissimilarity`
- Image URLs are from AWS S3, may need CORS handling
- Images may not exist for all materials (handle 404)

---

### GET /natural-language-summary

**Purpose:** Generate a human-readable, natural language description of a crystal structure using Robocrystallographer. Useful for explaining materials to non-experts.

**URL:** `/natural-language-summary`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `material_id` | string | Yes | Material ID (e.g., "mp-19770") |

**Response Format:**
```typescript
{
  material_id: string;
  description: string;    // Natural language description
}
```

**Example Request:**
```
GET /natural-language-summary?material_id=mp-19770
```

**Example Response:**
```json
{
  "material_id": "mp-19770",
  "description": "Fe2O3 is Corundum-structured and crystallizes in the trigonal R-3c space group. The structure is three-dimensional. Fe3+ is bonded to six O2- atoms to form FeO6 octahedra that share corners with six equivalent FeO6 octahedra and edges with three equivalent FeO6 octahedra. The corner-sharing octahedral tilt angles are 52°. All Fe–O bond lengths are 2.03 Å. O2- is bonded in a distorted T-shaped geometry to four equivalent Fe3+ atoms."
}
```

**Frontend Usage:**
- Display in a text card or expandable section
- Good for "About this material" or "Structure description" sections
- May take 2-5 seconds to generate (show loading state)
- Text can be quite long (multiple paragraphs)

**Example Fetch:**
```javascript
const response = await fetch(`http://localhost:8000/natural-language-summary?material_id=${materialId}`);
const result = await response.json();
// result.description is the text description
```

**Notes:**
- Uses Robocrystallographer library
- May fail for complex or unusual structures
- Response time varies (2-10 seconds typical)

---

### GET /predict

**Purpose:** Predict a material property using CrabNet neural network. This is the core ML prediction functionality.

**URL:** `/predict`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `formula` | string | Yes | Chemical formula (e.g., "Fe2O3", "AlLiSi") |
| `property_name` | string | Yes | Model name from `/get-models` |

**Response Format:**
```typescript
{
  formula: string;            // Original input formula
  normalized_formula: string; // Normalized for CrabNet (e.g., "Al1Li1Si1")
  property_name: string;      // Model used
  predicted_value: number;    // Predicted property value
  uncertainty: number;        // Model uncertainty estimate
  units: string;              // Units of predicted value
}
```

**Example Request:**
```
GET /predict?formula=Fe2O3&property_name=OQMD_Bandgap
```

**Example Response:**
```json
{
  "formula": "Fe2O3",
  "normalized_formula": "Fe2O3",
  "property_name": "OQMD_Bandgap",
  "predicted_value": 0.014567,
  "uncertainty": 0.167289,
  "units": "eV"
}
```

**Example with Formula Normalization:**
```
GET /predict?formula=AlLiSi&property_name=OQMD_Formation_Enthalpy
```

```json
{
  "formula": "AlLiSi",
  "normalized_formula": "Al1Li1Si1",
  "property_name": "OQMD_Formation_Enthalpy",
  "predicted_value": -0.428234,
  "uncertainty": 0.052891,
  "units": "eV/atom"
}
```

**Frontend Usage:**
- Primary prediction interface
- Display predicted_value with units
- Show uncertainty as ± value or confidence indicator
- Lower uncertainty = more confident prediction
- Formula normalization is automatic (user can input "AlLiSi" or "Al1Li1Si1")

**Example Fetch:**
```javascript
const response = await fetch(
  `http://localhost:8000/predict?formula=${encodeURIComponent(formula)}&property_name=${encodeURIComponent(selectedModel)}`
);
const prediction = await response.json();
// prediction has {formula, normalized_formula, property_name, predicted_value, uncertainty, units}
```

**Formula Input Flexibility:**
The API accepts formulas in multiple formats and normalizes them:

| Input | Normalized | Notes |
|-------|-----------|-------|
| `Fe2O3` | `Fe2O3` | Already normalized |
| `AlLiSi` | `Al1Li1Si1` | Adds explicit 1s |
| `H2O` | `H2O1` | Adds trailing 1 |
| `Ca(OH)2` | `Ca1H2O2` | Expands parentheses |
| `NaCl` | `Cl1Na1` | Sorted alphabetically |

**Notes:**
- First prediction with a new model may take 5-10 seconds (model loading)
- Subsequent predictions are faster (model cached)
- Uncertainty varies by model and composition
- Some compositions may give unexpected results if far from training data

---

### GET /health

**Purpose:** Health check endpoint for monitoring API status.

**URL:** `/health`

**Parameters:** None

**Response Format:**
```typescript
{
  status: string;           // "healthy"
  crabnet_loaded: boolean;  // Whether CrabNet is initialized
  models_available: number; // Number of available models
}
```

**Example Response:**
```json
{
  "status": "healthy",
  "crabnet_loaded": true,
  "models_available": 83
}
```

**Frontend Usage:**
- Check API availability on app load
- Display connection status indicator
- Retry logic if unhealthy

---

## Data Models

### ModelInfo
```typescript
interface ModelInfo {
  name: string;         // Unique identifier for model
  description: string;  // Human-readable description
  units: string;        // Units for predicted values
}
```

### AutocompleteResult
```typescript
interface AutocompleteResult {
  formula_pretty: string;  // Suggested formula
}
```

### MaterialBasicInfo
```typescript
interface MaterialBasicInfo {
  material_id: string;
  formula_pretty: string;
  volume: number | null;
  density: number | null;
  symmetry: {
    crystal_system: string;
    symbol: string;
    number: number;
    point_group: string;
  } | null;
}
```

### SimilarMaterial
```typescript
interface SimilarMaterial {
  material_id: string;
  formula: string;
  similarity: number;
  image_url: string;
}
```

### PredictionResult
```typescript
interface PredictionResult {
  formula: string;
  normalized_formula: string;
  property_name: string;
  predicted_value: number;
  uncertainty: number;
  units: string;
}
```

---

## Property Units Reference

### Band Gap Properties
| Model Name | Units | Description |
|------------|-------|-------------|
| `OQMD_Bandgap` | eV | Band gap from OQMD |
| `aflow__Egap` | eV | Band gap from AFLOW |
| `expt_gap0-4` | eV | Experimental band gap (5 models) |
| `mp_gap0-4` | eV | Materials Project band gap (5 models) |

### Formation/Energy Properties
| Model Name | Units | Description |
|------------|-------|-------------|
| `OQMD_Formation_Enthalpy` | eV/atom | Formation enthalpy |
| `OQMD_Energy_per_atom` | eV/atom | Total energy per atom |
| `aflow__energy_atom` | eV/atom | AFLOW energy per atom |
| `mp_e_form0-4` | eV/atom | MP formation energy (5 models) |
| `mp_e_hull` | eV/atom | Energy above hull |
| `CritExam__Ed` | eV/atom | Decomposition energy |
| `CritExam__Ef` | eV/atom | Formation energy |

### Mechanical Properties
| Model Name | Units | Description |
|------------|-------|-------------|
| `aflow__ael_bulk_modulus_vrh` | GPa | Bulk modulus |
| `aflow__ael_shear_modulus_vrh` | GPa | Shear modulus |
| `mp_bulk_modulus` | GPa | MP bulk modulus |
| `mp_shear_modulus` | GPa | MP shear modulus |
| `mp_elastic_anisotropy` | dimensionless | Elastic anisotropy |
| `elasticity_log10(G_VRH)0-4` | log10(GPa) | Log shear modulus |
| `elasticity_log10(K_VRH)0-4` | log10(GPa) | Log bulk modulus |
| `steels_yield0-4` | MPa | Steel yield strength |

### Thermal Properties
| Model Name | Units | Description |
|------------|-------|-------------|
| `aflow__agl_thermal_conductivity_300K` | W/(m·K) | Thermal conductivity |
| `aflow__agl_thermal_expansion_300K` | 1/K | Thermal expansion |
| `aflow__ael_debye_temperature` | K | Debye temperature |
| `phonons0-4` | cm⁻¹ | Phonon frequency |

### Other Properties
| Model Name | Units | Description |
|------------|-------|-------------|
| `OQMD_Volume_per_atom` | ų/atom | Volume per atom |
| `mp_mu_b` | μB/f.u. | Magnetic moment |
| `dielectric0-4` | dimensionless | Dielectric constant |
| `expt_is_metal0-4` | probability | Is metal (0-1) |
| `mp_is_metal0-4` | probability | Is metal (0-1) |
| `glass0-4` | probability | Glass forming ability |
| `castelli0-4` | eV | Perovskite stability |
| `jdft2d0-4` | meV/atom | 2D exfoliation energy |

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid formula or unknown model name |
| 404 | Not Found | Material ID not in database |
| 500 | Server Error | Prediction failed or internal error |
| 502 | Bad Gateway | Materials Project API unavailable |
| 503 | Service Unavailable | Missing dependencies (robocrys) |

### Error Response Format
```typescript
{
  detail: string;  // Error message
}
```

### Example Error Responses

**Invalid Formula (400):**
```json
{
  "detail": "Invalid formula: Unknown element: Xx"
}
```

**Unknown Model (400):**
```json
{
  "detail": "Invalid property name. Available models: OQMD_Bandgap, aflow__Egap... (use /get-models for full list)"
}
```

**Material Not Found (404):**
```json
{
  "detail": "Structure not found for mp-99999999"
}
```

**Materials Project API Error (502):**
```json
{
  "detail": "Materials Project API error: Connection timeout"
}
```

### Frontend Error Handling

```javascript
try {
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }
  const data = await response.json();
  // Handle success
} catch (error) {
  // Display error.message to user
  showError(error.message);
}
```

---

## Frontend Implementation Guide

### Recommended Tech Stack
- **Framework:** React, Vue, or Svelte
- **HTTP Client:** fetch API or axios
- **3D Visualization:** Three.js, react-three-fiber, or 3Dmol.js
- **UI Components:** Material UI, Chakra UI, or Tailwind CSS

### Suggested Page Structure

```
App
├── Header
│   └── Logo, Navigation
├── Main Content
│   ├── PropertySelector (dropdown from /get-models)
│   ├── FormulaSearch
│   │   ├── Input with autocomplete (/autocomplete-search)
│   │   └── Search button → /retrieve
│   ├── SearchResults
│   │   └── List/Grid of materials from /retrieve
│   └── MaterialDetail (when material selected)
│       ├── BasicInfo (from /summary)
│       ├── CrabNetPrediction (/predict)
│       ├── StructureViewer (3D from /summary structure data)
│       ├── NaturalLanguageDescription (/natural-language-summary)
│       └── RelatedMaterials (/related-materials)
└── Footer
```

### API Call Sequence

**User Flow 1: Search and Explore**
```
1. Page Load
   → GET /get-models (populate property dropdown)
   → GET /health (check API status)

2. User types formula
   → GET /autocomplete-search?formula=... (debounced)

3. User selects/submits formula
   → GET /retrieve?formula=...

4. User clicks a material
   → GET /summary?material_id=...
   → GET /predict?formula=...&property_name=... (selected property)
   → GET /related-materials?material_id=...
   → GET /natural-language-summary?material_id=... (lazy load)
```

### State Management

```typescript
interface AppState {
  // Models
  models: ModelInfo[];
  selectedModel: string;
  
  // Search
  searchQuery: string;
  autocompleteResults: AutocompleteResult[];
  searchResults: MaterialBasicInfo[];
  
  // Selected Material
  selectedMaterialId: string | null;
  materialSummary: MaterialSummary | null;
  prediction: PredictionResult | null;
  relatedMaterials: SimilarMaterial[];
  nlDescription: string | null;
  
  // UI State
  isLoading: boolean;
  error: string | null;
}
```

### Component Examples

**PropertySelector:**
```jsx
function PropertySelector({ models, selected, onChange }) {
  return (
    <select value={selected} onChange={e => onChange(e.target.value)}>
      {models.map(model => (
        <option key={model.name} value={model.name}>
          {model.description} ({model.units})
        </option>
      ))}
    </select>
  );
}
```

**FormulaSearchInput:**
```jsx
function FormulaSearchInput({ onSearch }) {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  
  // Debounced autocomplete
  useEffect(() => {
    const timer = setTimeout(async () => {
      if (query.length >= 2) {
        const res = await fetch(`/autocomplete-search?formula=${query}`);
        setSuggestions(await res.json());
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);
  
  return (
    <div>
      <input 
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Enter formula (e.g., Fe2O3)"
      />
      {suggestions.length > 0 && (
        <ul className="suggestions">
          {suggestions.map(s => (
            <li key={s.formula_pretty} onClick={() => {
              setQuery(s.formula_pretty);
              onSearch(s.formula_pretty);
            }}>
              {s.formula_pretty}
            </li>
          ))}
        </ul>
      )}
      <button onClick={() => onSearch(query)}>Search</button>
    </div>
  );
}
```

**PredictionDisplay:**
```jsx
function PredictionDisplay({ prediction }) {
  if (!prediction) return null;
  
  return (
    <div className="prediction-card">
      <h3>CrabNet Prediction</h3>
      <div className="property">{prediction.property_name}</div>
      <div className="value">
        {prediction.predicted_value.toFixed(4)} {prediction.units}
      </div>
      <div className="uncertainty">
        ± {prediction.uncertainty.toFixed(4)} {prediction.units}
      </div>
      <div className="formula-note">
        Input: {prediction.formula} → {prediction.normalized_formula}
      </div>
    </div>
  );
}
```

### 3D Structure Visualization

The `/summary` endpoint returns structure data that can be rendered in 3D:

```javascript
// Extract atom positions from summary
const atoms = summary.structure.sites.map(site => ({
  element: site.species[0].element,
  x: site.xyz[0],
  y: site.xyz[1],
  z: site.xyz[2]
}));

// Extract lattice parameters
const lattice = summary.structure.lattice;

// Use with Three.js, 3Dmol.js, or similar library
```

**Recommended 3D Libraries:**
- [3Dmol.js](https://3dmol.org/) - Specifically for molecular visualization
- [react-three-fiber](https://github.com/pmndrs/react-three-fiber) - React + Three.js
- [NGL Viewer](http://nglviewer.org/) - Molecular visualization

---

## Example User Flows

### Flow 1: Predict Band Gap for a Known Material

```
1. User sees property dropdown with "Band gap energy from OQMD database (eV)"
2. User types "TiO2" in search box
3. Autocomplete shows: TiO2, Ti2O3, TiO, Ti3O5...
4. User selects "TiO2"
5. User clicks Search
6. Results show multiple TiO2 polymorphs (anatase, rutile, brookite)
7. User clicks "mp-2657" (anatase)
8. Detail page shows:
   - Basic info: Tetragonal, I41/amd, density 3.89 g/cm³
   - CrabNet prediction: Band gap = 2.34 eV ± 0.15 eV
   - 3D structure visualization
   - Description: "TiO2 is Anatase-structured and crystallizes..."
   - Related materials: SnO2, ZrO2, HfO2...
```

### Flow 2: Explore a Novel Composition

```
1. User selects "Formation enthalpy per atom (eV/atom)"
2. User types "LiNiMnCoO" (arbitrary composition)
3. No autocomplete matches (not in MP database)
4. User clicks Search anyway
5. No results from Materials Project
6. System shows: "No materials found, but you can still predict!"
7. CrabNet predicts: Formation enthalpy = -1.23 eV/atom ± 0.08 eV/atom
8. User can explore what-if scenarios with different compositions
```

### Flow 3: Compare Similar Materials

```
1. User searches for "Fe2O3"
2. Clicks on mp-19770 (hematite)
3. Views full details and prediction
4. Scrolls to "Related Materials" section
5. Sees Ga2O3, Al2O3, Cr2O3 with similarity scores
6. Clicks on Al2O3 to compare
7. Can go back and forth comparing properties
```

---

## Rate Limiting & Performance Notes

- **Autocomplete:** Debounce 300-500ms to avoid excessive API calls
- **Materials Project:** External API, may have rate limits
- **CrabNet Predictions:** First prediction ~5-10s (model loading), subsequent ~0.5-2s
- **Natural Language Summary:** 2-10 seconds (complex calculation)
- **Structure Images:** Hosted on AWS S3, generally fast

### Caching Recommendations

```javascript
// Client-side caching suggestions
const cache = {
  models: null,           // Cache indefinitely (doesn't change)
  summaries: new Map(),   // Cache by material_id
  predictions: new Map(), // Cache by formula+model
  related: new Map(),     // Cache by material_id
};
```

---

## CORS Configuration

The API has CORS enabled for all origins (`*`). For production, the frontend URL should be whitelisted in the API's CORS configuration.

Current configuration allows:
- All origins
- All HTTP methods
- All headers
- Credentials

---

## Contact & Support

- **API Base URL:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

This documentation should provide all necessary information to build a complete frontend application that interacts with the CrabNet Material Property Prediction API.
