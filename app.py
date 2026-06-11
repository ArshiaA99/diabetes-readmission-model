from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Dict, Any, Optional

from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.data_processor import DataProcessor

# Global storage for model artifacts to avoid loading them on every request
artifacts: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to handle API startup and shutdown tasks smoothly."""
    models_dir = Path("models")
    model_path = models_dir / "diabetes_model.cbm"
    scaler_path = models_dir / "scaler.joblib"
    features_path = models_dir / "feature_names.json"

    # Verify everything exists before spinning up the server
    if not (model_path.exists() and scaler_path.exists() and features_path.exists()):
        raise RuntimeError(
            "Required model artifacts are missing. Please run main.py to train and save the model first."
        )

    print("🩺 Loading GlucoGuard AI clinical model artifacts...")
    artifacts["model"] = CatBoostClassifier().load_model(str(model_path))
    artifacts["scaler"] = joblib.load(scaler_path)
    with open(features_path, "r") as file:
        artifacts["feature_names"] = json.load(file)
    
    # Instance of DataProcessor for feature mapping pipelines
    artifacts["processor"] = DataProcessor()
    print("🚀 GlucoGuard AI is fully operational.")
    yield
    artifacts.clear()


# Theme-related API Metadata
app = FastAPI(
    title="🩸 GlucoGuard AI — Clinical Decision Support System",
    description=(
        "An advanced, modern REST API designed to evaluate the 30-day readmission risk "
        "of diabetic patients. Developed for hospital triage integration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PatientPayload(BaseModel):
    """Schema representing clinical features expected for validation."""
    race: str = Field(default="Caucasian", example="Caucasian")
    gender: str = Field(default="Female", example="Female")
    age: str = Field(default="[70-80)", example="[70-80)", description="Age group format: [Min-Max)")
    admission_type_id: int = Field(default=1, example=1)
    discharge_disposition_id: int = Field(default=1, example=1)
    admission_source_id: int = Field(default=7, example=7)
    time_in_hospital: int = Field(default=3, example=3, description="Duration of stay in days")
    num_lab_procedures: int = Field(default=40, example=42)
    num_procedures: int = Field(default=0, example=1)
    num_medications: int = Field(default=15, example=18)
    number_outpatient: int = Field(default=0, example=0)
    number_emergency: int = Field(default=0, example=0)
    number_inpatient: int = Field(default=0, example=1)
    diag_1: str = Field(default="428", example="428", description="Primary ICD-9 diagnosis code")
    number_diagnoses: int = Field(default=9, example=9)
    max_glu_serum: str = Field(default="None", example="None")
    A1Cresult: str = Field(default="None", example=">8")
    metformin: str = Field(default="No", example="No")
    insulin: str = Field(default="No", example="Yes")
    diabetesMed: str = Field(default="No", example="Yes")


@app.get("/", tags=["System Health"])
async def root():
    """Styled dashboard summary of the API system."""
    return {
        "system": "GlucoGuard AI (Clinical Decision Support)",
        "status": "Online",
        "api_docs": "/docs",
        "model_format": "CatBoost Native Binary (.cbm)",
        "supported_features_count": len(artifacts.get("feature_names", []))
    }


@app.post("/api/v1/predict", tags=["Clinical Analytics"])
async def predict_readmission(payload: PatientPayload):
    """Processes raw incoming clinical attributes and returns a readmission risk stratification."""
    try:
        # Convert Pydantic payload directly to DataFrame format
        raw_data = pd.DataFrame([payload.model_dump()])

        processor = artifacts["processor"]
        scaler = artifacts["scaler"]
        model = artifacts["model"]
        expected_features = artifacts["feature_names"]

        # Run incoming data through data cleaning and categorical feature engineering
        cleaned_df = processor.clean_data(raw_data)
        engineered_df = processor.engineer_features(cleaned_df)

        # Dynamic feature alignment: Fills missing one-hot variables with 0s
        for col in expected_features:
            if col not in engineered_df.columns:
                engineered_df[col] = 0

        # Enforce identical column sequencing as observed during training
        aligned_df = engineered_df[expected_features]

        # Apply standard scaling transformations and score models
        scaled_features = scaler.transform(aligned_df.values)
        probability = float(model.predict_proba(scaled_features)[0][1])
        prediction = int(probability > 0.5)

        # Map metrics to clinical tier parameters
        risk_tier = "High Risk" if probability >= 0.7 else "Moderate Risk" if probability >= 0.4 else "Low Risk"

        return {
            "patient_summary": {
                "age_group": payload.age,
                "primary_icd9": payload.diag_1,
                "days_hospitalized": payload.time_in_hospital
            },
            "analysis": {
                "readmission_predicted": bool(prediction),
                "readmission_probability": round(probability, 4),
                "risk_stratification": risk_tier,
            },
            "clinical_guidance": (
                "Prioritize case management follow-up within 48 hours of discharge."
                if prediction else "Standard discharge protocol recommended."
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while evaluating clinical features: {str(e)}"
        )
