from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
import pickle

# CHANGE THESE TWO LINES:
from model import DroughtPredictorGRU
from route import router

MODEL_PATH = "../drought_gru_model.pth"
SCALER_PATH = "../scaler.pkl"

# Lifespan manager: Loads the model once when the API boots up
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing system resources... Loading PyTorch model.")
    try:
        # Initialize the model blueprint
        model_instance = DroughtPredictorGRU()
        # Load parameters onto standard CPU context
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model_instance.eval()
        
        # Inject the loaded model into global app state memory
        app.state.model = model_instance
        # Load scaler too
        with open(SCALER_PATH, "rb") as f:
            app.state.scaler = pickle.load(f)   # ← add this
            
        print("PyTorch weights successfully loaded into memory state.")
    except FileNotFoundError:
        print(f"CRITICAL: System could not locate weights at '{MODEL_PATH}'.")
    
    yield
    print("Shutting down resources...")

# Initialize the main API application instance
app = FastAPI(
    title="Drought Prediction Production API",
    description="Clean asynchronous 3-tier endpoint layout.",
    version="3.0.0",
    lifespan=lifespan
)

# Connect your clean routes directly to the core application
app.include_router(router, tags=["Inference Engine"])