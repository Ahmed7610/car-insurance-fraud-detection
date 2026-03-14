# =============================================================================
# main.py  —  Car Insurance Fraud Detection API
# =============================================================================
#
# WHAT IS FastAPI?
# ─────────────────
# FastAPI is a Python framework for building APIs (Application Programming
# Interfaces). An API is basically a "door" that lets OTHER programs talk to
# your model.
#
# For example:
#   - A mobile app sends claim data → your API returns "FRAUD" or "LEGIT"
#   - A website form submits data  → your API returns a fraud probability
#
# HOW DOES IT WORK?
# ──────────────────
# You define "routes" (URLs) that respond to HTTP requests:
#   GET  /         → health check (is the server alive?)
#   POST /predict  → send claim data, get back a fraud prediction
#
# WHY FastAPI SPECIFICALLY?
# ──────────────────────────
# - Auto-generates interactive documentation at /docs (try it!)
# - Validates request data automatically using Pydantic
# - Very fast and beginner-friendly
#
# HOW TO RUN THIS FILE:
# ──────────────────────
# 1. Install dependencies:
#      pip install fastapi uvicorn scikit-learn
#
# 2. Put your .pkl files in a folder called model_artifacts/
#
# 3. Run the server:
#      uvicorn main:app --reload
#
# 4. Open browser at:
#      http://127.0.0.1:8000/docs   ← interactive API explorer
# =============================================================================

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional


# =============================================================================
# STEP 1: Load the saved model artifacts
# =============================================================================
# We load all 4 pickle files once when the server starts.
# Loading once (not on every request) keeps the API fast.


def load_artifact(path: str):
    """Helper function to load a single pickle file."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not find '{path}'. "
            "Make sure you ran the notebook and placed the .pkl files "
            "in the model_artifacts/ folder."
        )


# ✅ CORRECT — variable names match file names
model = load_artifact(
    "/home/ahmed/Downloads/Depi/fraud_project/model_artifacts/model.pkl"
)
scaler = load_artifact(
    "/home/ahmed/Downloads/Depi/fraud_project/model_artifacts/scaler.pkl"
)
label_encoders = load_artifact(
    "/home/ahmed/Downloads/Depi/fraud_project/model_artifacts/label_encoders.pkl"
)
feature_columns = load_artifact(
    "/home/ahmed/Downloads/Depi/fraud_project/model_artifacts/feature_columns.pkl"
)
print("✅ Model and artifacts loaded successfully!")


# =============================================================================
# STEP 2: Create the FastAPI app
# =============================================================================
# Think of 'app' as the main engine of your API.
# All routes (URLs) are registered on this object.

app = FastAPI(
    title="🚗 Car Insurance Fraud Detection API",
    description=(
        "Send insurance claim details and get back a fraud prediction. "
        "Built with FastAPI + scikit-learn Random Forest."
    ),
    version="1.0.0",
)


# =============================================================================
# STEP 3: Define the Input Schema with Pydantic
# =============================================================================
# Pydantic's BaseModel defines EXACTLY what JSON data the API expects.
# If a required field is missing or has the wrong type, FastAPI automatically
# returns a clear error message — you don't have to write that validation yourself.
#
# Field(...) = required field
# Field(default=...) = optional field with a default value
# The descriptions show up in the auto-generated /docs page.


class ClaimInput(BaseModel):
    policy_state: str = Field(
        ..., description="State where policy was issued, e.g. 'CA'"
    )
    policy_deductible: int = Field(..., description="Deductible amount, e.g. 500")
    policy_annual_premium: float = Field(
        ..., description="Annual premium amount, e.g. 1200.50"
    )
    insured_age: int = Field(..., description="Age of the insured person, e.g. 35")
    insured_sex: str = Field(..., description="'MALE', 'FEMALE', or 'OTHER'")
    insured_education_level: str = Field(
        ..., description="Education level, e.g. 'College'"
    )
    insured_occupation: str = Field(..., description="Occupation, e.g. 'Doctor'")
    insured_hobbies: str = Field(..., description="Hobby, e.g. 'chess'")
    incident_type: str = Field(
        ..., description="Type of incident, e.g. 'Vehicle Theft'"
    )
    collision_type: str = Field(..., description="Collision type, e.g. 'Rear'")
    incident_severity: str = Field(..., description="Severity level, e.g. 'Total Loss'")
    authorities_contacted: str = Field(
        ..., description="Who was contacted, e.g. 'Police'"
    )
    incident_state: str = Field(
        ..., description="State where incident happened, e.g. 'OH'"
    )
    incident_hour_of_the_day: int = Field(..., description="Hour of incident (0–23)")
    number_of_vehicles_involved: int = Field(
        ..., description="How many vehicles were involved"
    )
    bodily_injuries: int = Field(..., description="Number of bodily injuries")
    witnesses: int = Field(..., description="Number of witnesses")
    police_report_available: str = Field(..., description="'Yes' or 'No'")
    claim_amount: float = Field(..., description="Claim amount in dollars")
    total_claim_amount: float = Field(..., description="Total claim amount in dollars")

    # ── Example payload shown in /docs ────────────────────────────────────────
    class Config:
        json_schema_extra = {
            "example": {
                "policy_state": "CA",
                "policy_deductible": 500,
                "policy_annual_premium": 1350.75,
                "insured_age": 42,
                "insured_sex": "MALE",
                "insured_education_level": "College",
                "insured_occupation": "Manager",
                "insured_hobbies": "reading",
                "incident_type": "Multi-vehicle Collision",
                "collision_type": "Rear",
                "incident_severity": "Total Loss",
                "authorities_contacted": "Police",
                "incident_state": "OH",
                "incident_hour_of_the_day": 14,
                "number_of_vehicles_involved": 2,
                "bodily_injuries": 1,
                "witnesses": 2,
                "police_report_available": "Yes",
                "claim_amount": 45000.00,
                "total_claim_amount": 50000.00,
            }
        }


# =============================================================================
# STEP 4: Define the Output Schema
# =============================================================================


class PredictionOutput(BaseModel):
    prediction: str  # "FRAUD" or "LEGITIMATE"
    fraud_probability: float  # Probability between 0 and 1
    confidence: str  # Human-readable confidence level


# =============================================================================
# STEP 5: Define the API Routes
# =============================================================================


@app.get("/", tags=["Health Check"])
def root():
    """
    Health check endpoint.
    If the server is running, this returns a simple 'alive' message.
    Useful for deployment monitoring (e.g. Kubernetes, Docker health checks).
    """
    return {
        "status": "online",
        "message": "Car Insurance Fraud Detection API is running.",
        "docs": "Visit /docs for the interactive API explorer.",
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(claim: ClaimInput):
    """
    ## Predict Fraud for an Insurance Claim

    Send the claim details as JSON and receive:
    - **prediction**: FRAUD or LEGITIMATE
    - **fraud_probability**: the model's confidence (0.0 to 1.0)
    - **confidence**: LOW / MEDIUM / HIGH label

    ### How it works internally:
    1. Input data is received as JSON
    2. Categorical fields are encoded using saved LabelEncoders
    3. All features are scaled using the saved StandardScaler
    4. The trained model predicts fraud probability
    5. Result is returned as JSON
    """

    # ── Convert Pydantic model to a plain Python dict ──────────────────────────
    input_data = claim.dict()

    # ── Encode categorical columns ────────────────────────────────────────────
    # We use the SAME LabelEncoders that were fitted during training.
    # This guarantees that 'Police' maps to the same number it did during training.
    for col, le in label_encoders.items():
        if col in input_data:
            raw_value = input_data[col]

            # Handle unseen categories gracefully
            if raw_value not in le.classes_:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown value '{raw_value}' for field '{col}'. "
                    f"Valid options are: {list(le.classes_)}",
                )

            input_data[col] = int(le.transform([raw_value])[0])

    # ── Build the feature vector in the correct column order ──────────────────
    # The model was trained with columns in a specific order.
    # We MUST pass them in the same order — otherwise results will be wrong!
    try:
        feature_vector = np.array([[input_data[col] for col in feature_columns]])
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    # ── Scale the features ────────────────────────────────────────────────────
    feature_vector_scaled = scaler.transform(feature_vector)

    # ── Make prediction ───────────────────────────────────────────────────────
    prediction_label = model.predict(feature_vector_scaled)[0]  # 0 or 1
    prediction_proba = model.predict_proba(feature_vector_scaled)[0][
        1
    ]  # fraud probability

    # ── Format the output ─────────────────────────────────────────────────────
    result_label = "FRAUD" if prediction_label == 1 else "LEGITIMATE"

    if prediction_proba >= 0.75:
        confidence = "HIGH"
    elif prediction_proba >= 0.50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return PredictionOutput(
        prediction=result_label,
        fraud_probability=round(float(prediction_proba), 4),
        confidence=confidence,
    )


@app.get("/model-info", tags=["Info"])
def model_info():
    """Returns information about the loaded model."""
    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "categorical_cols": list(label_encoders.keys()),
    }
