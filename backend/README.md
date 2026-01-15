# CrabNet Material Property Prediction API

A FastAPI server for predicting material properties using CrabNet and retrieving material information from the Materials Project API.

## Features

- **CrabNet Property Prediction**: Predict 83+ material properties from chemical formulas
- **Materials Project Integration**: Autocomplete, search, and retrieve material data
- **Robocrystallographer**: Generate natural language descriptions of crystal structures
- **Related Materials**: Find similar materials with structure images

## Setup

### 1. Install Dependencies

```powershell
# Activate virtual environment
cd d:\Repositories\CrabNet
.venv\scripts\activate

# Install API dependencies
pip install fastapi uvicorn httpx robocrys mp-api
```

### 2. Configure API Key

Edit `config.py` and set your Materials Project API key:

```python
MP_API_KEY = "your-api-key-here"
```

Or set it as an environment variable:
```powershell
$env:MP_API_KEY = "your-api-key-here"
```

### 3. Run the Server

```powershell
cd API_backend
python main.py
```

Or using uvicorn directly:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `/get-models`
Get list of all available CrabNet models for property prediction.

**Response:**
```json
[
  {"name": "OQMD_Bandgap", "description": "Band gap energy from OQMD database", "units": "eV"},
  {"name": "aflow__ael_bulk_modulus_vrh", "description": "Bulk modulus (VRH average)", "units": "GPa"}
]
```

### `/autocomplete-search?formula=Fe2O`
Autocomplete chemical formula search.

**Parameters:**
- `formula`: Partial formula to search

**Response:**
```json
[
  {"formula_pretty": "Fe2OF2"},
  {"formula_pretty": "Fe2OF3"}
]
```

### `/retrieve?formula=Fe2O3`
Retrieve materials matching the given formula.

**Parameters:**
- `formula`: Chemical formula to search

**Response:**
```json
{
  "data": [
    {
      "formula_pretty": "Fe2O3",
      "volume": 982.25,
      "density": 4.319,
      "symmetry": {...},
      "material_id": "mp-1244869"
    }
  ]
}
```

### `/summary?material_id=mp-19770`
Get comprehensive summary of a material.

**Parameters:**
- `material_id`: Material ID (e.g., mp-19770)

**Response:** Full Materials Project summary data

### `/related-materials?material_id=mp-19770`
Get similar/related materials.

**Parameters:**
- `material_id`: Material ID

**Response:**
```json
[
  {
    "material_id": "mp-1243",
    "formula": "Ga4O6",
    "similarity": 96.24,
    "image_url": "https://materialsproject-build.s3.amazonaws.com/images/structures/mp-1243.png"
  }
]
```

### `/natural-language-summary?material_id=mp-19770`
Generate a natural language description of the crystal structure.

**Parameters:**
- `material_id`: Material ID

**Response:**
```json
{
  "material_id": "mp-19770",
  "description": "Fe2O3 is Corundum-like structured and crystallizes in the trigonal R-3c space group..."
}
```

### `/predict?formula=Fe2O3&property_name=OQMD_Bandgap`
Predict a material property using CrabNet.

**Parameters:**
- `formula`: Chemical formula (e.g., Fe2O3, AlLiSi)
- `property_name`: Property model name (e.g., OQMD_Bandgap)

**Response:**
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

### `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "crabnet_loaded": true,
  "models_available": 83
}
```

## Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Formula Normalization

The API automatically normalizes chemical formulas for CrabNet:

| Input | Normalized |
|-------|-----------|
| AlLiSi | Al1Li1Si1 |
| Fe2O3 | Fe2O3 |
| H2O | H2O1 |
| Ca(OH)2 | Ca1H2O2 |

## File Structure

```
API_backend/
├── main.py              # FastAPI application and routes
├── crabnet_predictor.py # CrabNet model loading and prediction
├── formula_utils.py     # Chemical formula parsing and normalization
├── config.py            # Configuration (API keys, etc.)
├── __init__.py          # Package init
└── README.md            # This file
```

## Available Properties

The API provides predictions for 83+ material properties including:

### Band Gap
- `OQMD_Bandgap` (eV)
- `aflow__Egap` (eV)

### Mechanical Properties
- `aflow__ael_bulk_modulus_vrh` (GPa)
- `aflow__ael_shear_modulus_vrh` (GPa)
- `mp_bulk_modulus` (GPa)
- `mp_shear_modulus` (GPa)

### Thermal Properties
- `aflow__agl_thermal_conductivity_300K` (W/(m·K))
- `aflow__agl_thermal_expansion_300K` (1/K)
- `aflow__ael_debye_temperature` (K)

### Formation/Energy
- `OQMD_Formation_Enthalpy` (eV/atom)
- `OQMD_Energy_per_atom` (eV/atom)
- `mp_e_hull` (eV/atom)

Use `/get-models` endpoint for the complete list.

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid formula, unknown model)
- `404`: Not found (material ID not in database)
- `500`: Server error (prediction failed)
- `502`: External API error (Materials Project unavailable)
- `503`: Service unavailable (missing dependencies)

## CORS

CORS is enabled for all origins by default. For production, configure appropriately in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    ...
)
```
