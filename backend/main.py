"""
CrabNet Material Property Prediction API

FastAPI server for predicting material properties using CrabNet models
and integrating with Materials Project API for additional information.
"""
import os
import sys
import re
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Add parent directory to path for CrabNet imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from crabnet_predictor import CrabNetPredictor
from formula_utils import normalize_formula
from config import MP_API_KEY, MP_BASE_URL

# Initialize FastAPI app
app = FastAPI(
    title="CrabNet Material Property Prediction API",
    description="API for predicting material properties using CrabNet and retrieving material information from Materials Project",
    version="1.0.0"
)

# Debug mode configuration
DEBUG_MODE = True  # Set to True to restrict available properties
ALLOWED_PROPERTIES = ["mp_gap0", "mp_bulk_modulus", "mp_e_form0", "mp_shear_modulus", "OQMD_Volume_per_atom", "mp_mu_b", "dielectric0", "OQMD_Energy_per_atom", "mp_is_metal0"]

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CrabNet predictor (singleton)
predictor = CrabNetPredictor()


# ============================================================================
# Response Models
# ============================================================================

class ModelInfo(BaseModel):
    name: str
    description: str
    units: str


class AutocompleteResult(BaseModel):
    formula_pretty: str


class MaterialBasicInfo(BaseModel):
    material_id: str
    formula_pretty: str
    volume: Optional[float] = None
    density: Optional[float] = None
    symmetry: Optional[dict] = None


class SimilarMaterial(BaseModel):
    material_id: str
    formula: str
    similarity: float
    image_url: str


class PredictionResult(BaseModel):
    formula: str
    normalized_formula: str
    property_name: str
    predicted_value: float
    uncertainty: float
    units: str


# ============================================================================
# Property Units Mapping
# ============================================================================

PROPERTY_UNITS = {
    # Band gaps
    "OQMD_Bandgap": "eV",
    "aflow__Egap": "eV",
    "expt_gap": "eV",
    "mp_gap": "eV",
    
    # Formation/Energy
    "OQMD_Formation_Enthalpy": "eV/atom",
    "OQMD_Energy_per_atom": "eV/atom",
    "aflow__energy_atom": "eV/atom",
    "mp_e_form": "eV/atom",
    "mp_e_hull": "eV/atom",
    "CritExam__Ed": "eV/atom",
    "CritExam__Ef": "eV/atom",
    
    # Mechanical
    "aflow__ael_bulk_modulus_vrh": "GPa",
    "aflow__ael_shear_modulus_vrh": "GPa",
    "mp_bulk_modulus": "GPa",
    "mp_shear_modulus": "GPa",
    "mp_elastic_anisotropy": "dimensionless",
    "elasticity_log10(G_VRH)": "log10(GPa)",
    "elasticity_log10(K_VRH)": "log10(GPa)",
    
    # Thermal
    "aflow__agl_thermal_conductivity_300K": "W/(m·K)",
    "aflow__agl_thermal_expansion_300K": "1/K",
    "aflow__ael_debye_temperature": "K",
    "phonons": "cm⁻¹",
    
    # Volume
    "OQMD_Volume_per_atom": "Å³/atom",
    
    # Magnetic
    "mp_mu_b": "μB/f.u.",
    
    # Classification
    "expt_is_metal": "probability",
    "mp_is_metal": "probability",
    "glass": "probability",
    
    # Dielectric
    "dielectric": "dimensionless",
    
    # Other
    "castelli": "eV",
    "jdft2d": "meV/atom",
    "steels_yield": "MPa",
}


PROPERTY_DESCRIPTIONS = {
    "OQMD_Bandgap": "Band gap energy from OQMD database",
    "aflow__Egap": "Band gap energy from AFLOW database",
    "OQMD_Formation_Enthalpy": "Formation enthalpy per atom",
    "OQMD_Energy_per_atom": "Total energy per atom",
    "aflow__energy_atom": "Energy per atom from AFLOW",
    "aflow__ael_bulk_modulus_vrh": "Bulk modulus (VRH average)",
    "aflow__ael_shear_modulus_vrh": "Shear modulus (VRH average)",
    "aflow__agl_thermal_conductivity_300K": "Thermal conductivity at 300K",
    "aflow__agl_thermal_expansion_300K": "Thermal expansion coefficient at 300K",
    "aflow__ael_debye_temperature": "Debye temperature",
    "mp_bulk_modulus": "Bulk modulus from Materials Project",
    "mp_shear_modulus": "Shear modulus from Materials Project",
    "mp_e_hull": "Energy above convex hull",
    "mp_elastic_anisotropy": "Elastic anisotropy index",
    "mp_mu_b": "Magnetic moment per formula unit",
    "OQMD_Volume_per_atom": "Volume per atom",
    "CritExam__Ed": "Decomposition energy",
    "CritExam__Ef": "Formation energy",
    "expt_gap": "Experimental band gap",
    "mp_gap": "Band gap from Materials Project",
    "expt_is_metal": "Metallic classification (experimental)",
    "mp_is_metal": "Metallic classification (MP)",
    "mp_e_form": "Formation energy from Materials Project",
    "dielectric": "Dielectric constant",
    "elasticity_log10(G_VRH)": "Log10 of shear modulus",
    "elasticity_log10(K_VRH)": "Log10 of bulk modulus",
    "phonons": "Phonon frequency",
    "glass": "Glass forming ability",
    "castelli": "Perovskite stability (Castelli)",
    "jdft2d": "2D material exfoliation energy",
    "steels_yield": "Steel yield strength",
}


def get_units(model_name: str) -> str:
    """Get units for a model, handling numbered variants like 'mp_gap0'"""
    # Try exact match first
    if model_name in PROPERTY_UNITS:
        return PROPERTY_UNITS[model_name]
    
    # Try removing trailing numbers
    base_name = re.sub(r'\d+$', '', model_name)
    if base_name in PROPERTY_UNITS:
        return PROPERTY_UNITS[base_name]
    
    return "units"


def get_description(model_name: str) -> str:
    """Get description for a model"""
    if model_name in PROPERTY_DESCRIPTIONS:
        return PROPERTY_DESCRIPTIONS[model_name]
    
    base_name = re.sub(r'\d+$', '', model_name)
    if base_name in PROPERTY_DESCRIPTIONS:
        return PROPERTY_DESCRIPTIONS[base_name]
    
    return f"Predicted {model_name} property"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "CrabNet Material Property Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/get-models",
            "/autocomplete-search",
            "/retrieve",
            "/summary",
            "/related-materials",
            "/natural-language-summary",
            "/predict"
        ]
    }


@app.get("/get-models", response_model=List[ModelInfo])
async def get_models():
    """
    Get list of all available CrabNet models for property prediction.
    
    Returns a list of model names with descriptions and units.
    """
    models = predictor.list_models()
    
    if DEBUG_MODE:
        models = [m for m in models if m in ALLOWED_PROPERTIES]
    
    result = []
    for model_name in models:
        result.append(ModelInfo(
            name=model_name,
            description=get_description(model_name),
            units=get_units(model_name)
        ))
    
    return result


@app.get("/autocomplete-search", response_model=List[AutocompleteResult])
async def autocomplete_search(formula: str = Query(..., description="Partial formula to search")):
    """
    Autocomplete chemical formula search using Materials Project API.
    
    Returns up to 10 matching formulas.
    """
    if not formula or len(formula) < 1:
        return []
    
    url = f"{MP_BASE_URL}/materials/core/formula_autocomplete/"
    params = {
        "formula": formula,
        "limit": 10
    }
    headers = {
        "accept": "application/json",
        "X-API-KEY": MP_API_KEY
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            return [AutocompleteResult(formula_pretty=item["formula_pretty"]) 
                    for item in data.get("data", [])]
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Materials Project API error: {str(e)}")


@app.get("/retrieve")
async def retrieve_materials(formula: str = Query(..., description="Chemical formula to search")):
    """
    Retrieve materials matching the given formula from Materials Project.
    
    Returns basic information about matching materials including structure data.
    """
    url = f"{MP_BASE_URL}/materials/core/"
    params = {
        "formula": formula,
        "deprecated": "false",
        "_per_page": 10,
        "_skip": 0,
        "_limit": 10,
        "_fields": "material_id,formula_pretty,volume,density,symmetry",
        "_all_fields": "false",
        "license": "All"
    }
    headers = {
        "accept": "application/json",
        "X-API-KEY": MP_API_KEY
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Materials Project API error: {str(e)}")


@app.get("/summary")
async def get_summary(material_id: str = Query(..., description="Material ID (e.g., mp-19770)")):
    """
    Get comprehensive summary of a material from Materials Project.
    
    Returns all available fields for the specified material.
    """
    url = f"{MP_BASE_URL}/materials/summary/"
    params = {
        "material_ids": material_id,
        "deprecated": "false",
        "_per_page": 1,
        "_skip": 0,
        "_limit": 1,
        "_all_fields": "true",
        "license": "BY-C"
    }
    headers = {
        "accept": "application/json",
        "X-API-KEY": MP_API_KEY
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Materials Project API error: {str(e)}")


@app.get("/related-materials", response_model=List[SimilarMaterial])
async def get_related_materials(material_id: str = Query(..., description="Material ID (e.g., mp-19770)")):
    """
    Get similar/related materials for a given material ID.
    
    Returns top 5 similar materials with similarity scores and structure images.
    """
    url = f"{MP_BASE_URL}/materials/similarity/"
    params = {
        "material_ids": material_id,
        "_per_page": 5,
        "_skip": 0,
        "_limit": 5,
        "_fields": "formula_pretty,material_id,sim",
        "_all_fields": "false"
    }
    headers = {
        "accept": "application/json",
        "X-API-KEY": MP_API_KEY
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract similar materials from the response
            if data.get("data") and len(data["data"]) > 0:
                sim_list = data["data"][0].get("sim", [])
                
                # Take top 5 similar materials
                for sim_item in sim_list[:5]:
                    task_id = sim_item.get("task_id", "")
                    dissimilarity = sim_item.get("dissimilarity", 0)
                    similarity = max(0, 100 - dissimilarity)  # Convert to similarity
                    formula = sim_item.get("formula", "")
                    
                    # Clean up formula (remove spaces)
                    formula_clean = formula.replace(" ", "")
                    
                    results.append(SimilarMaterial(
                        material_id=task_id,
                        formula=formula_clean,
                        similarity=round(similarity, 2),
                        image_url=f"https://materialsproject-build.s3.amazonaws.com/images/structures/{task_id}.png"
                    ))
            
            return results
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Materials Project API error: {str(e)}")


@app.get("/natural-language-summary")
async def get_natural_language_summary(material_id: str = Query(..., description="Material ID (e.g., mp-19770)")):
    """
    Generate a natural language description of a crystal structure using Robocrystallographer.
    
    Returns a human-readable description of the material's structure.
    """
    try:
        from robocrys import StructureCondenser, StructureDescriber
        from mp_api.client import MPRester
        
        # Get structure from Materials Project
        with MPRester(api_key=MP_API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(material_id)
        
        if structure is None:
            raise HTTPException(status_code=404, detail=f"Structure not found for {material_id}")
        
        # Generate description using robocrystallographer
        condenser = StructureCondenser()
        describer = StructureDescriber()
        
        condensed_structure = condenser.condense_structure(structure)
        description = describer.describe(condensed_structure)
        
        return {
            "material_id": material_id,
            "description": description
        }
    
    except ImportError:
        raise HTTPException(
            status_code=503, 
            detail="Robocrystallographer not installed. Install with: pip install robocrys mp-api"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating description: {str(e)}")


@app.get("/predict", response_model=PredictionResult)
async def predict_property(
    formula: str = Query(..., description="Chemical formula (e.g., Fe2O3, AlLiSi)"),
    property_name: str = Query(..., description="Property model name (e.g., OQMD_Bandgap)")
):
    """
    Predict a material property using CrabNet.
    
    Accepts chemical formulas in various formats and normalizes them for CrabNet.
    Returns predicted value with uncertainty and units.
    """
    # Validate property name
    available_models = predictor.list_models()
    if property_name not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid property name. Available models: {available_models[:10]}... (use /get-models for full list)"
        )
    
    # Normalize formula for CrabNet
    try:
        normalized = normalize_formula(formula)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid formula: {str(e)}")
    
    # Run prediction
    try:
        predicted_value, uncertainty = predictor.predict(normalized, property_name)
        
        return PredictionResult(
            formula=formula,
            normalized_formula=normalized,
            property_name=property_name,
            predicted_value=round(predicted_value, 6),
            uncertainty=round(uncertainty, 6),
            units=get_units(property_name)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "crabnet_loaded": predictor.is_initialized,
        "models_available": len(predictor.list_models())
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
