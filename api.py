"""HTTP interface for scoring one customer with a saved churn model."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churn.data import FEATURES
from churn.modeling import ARTIFACT_VERSION, predict_frame


class Customer(BaseModel):
    """One raw customer record, before the saved pipeline cleans its features."""

    model_config = ConfigDict(extra="forbid")

    customerID: str | None = None
    gender: str
    SeniorCitizen: Literal[0, 1]
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(ge=0)


class Prediction(BaseModel):
    customerID: str | None
    churn_score: float
    predicted_churn: Literal["Yes", "No"]
    decision_threshold: float
    model_name: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("CHURN_MODEL_PATH")
    if not model_path:
        raise RuntimeError("Set CHURN_MODEL_PATH to a trusted churn_model.joblib file.")
    # Loading a joblib file can execute code. Only use a model trained by this project.
    bundle = joblib.load(Path(model_path))
    if (bundle.get("artifact_version") != ARTIFACT_VERSION
            or bundle.get("feature_columns") != list(FEATURES)):
        raise RuntimeError("Model artifact version or feature schema does not match this API.")
    app.state.model_bundle = bundle
    yield


app = FastAPI(title="Telco Churn API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the web process is running and can answer requests."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    """Apply the saved preprocessing, fitted model, and decision threshold."""
    frame = pd.DataFrame([customer.model_dump()])
    try:
        result = predict_frame(app.state.model_bundle, frame).iloc[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    bundle = app.state.model_bundle
    return Prediction(
        customerID=customer.customerID,
        churn_score=float(result["churn_score"]),
        predicted_churn=result["predicted_churn"],
        decision_threshold=float(bundle["threshold"]),
        model_name=bundle["model_name"],
    )
