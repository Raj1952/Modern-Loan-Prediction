from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from .model import LoanSimulatorModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "loan_approval_dataset.csv")
simulator = LoanSimulatorModel(data_path)

@app.on_event("startup")
def startup_event():
    # Train the model initially
    simulator.train()

@app.get("/api/train")
def get_training_info():
    return simulator.train()

class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int

@app.post("/api/predict")
def predict(application: LoanApplication):
    # Convert Pydantic model to dict
    app_data = application.dict()
    result = simulator.predict_explainable(app_data)
    return result
