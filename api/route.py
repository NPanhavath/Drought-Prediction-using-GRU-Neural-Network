from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import torch
import numpy as np

# Create an independent router object
router = APIRouter()

# --- Request Body Validation Schemas ---
class DailyMetrics(BaseModel):
    PS: float = Field(default=0.0, description="Surface Pressure")
    elevation: float = Field(default=0.0, description="Elevation")
    T2M_MAX: float = Field(default=0.0, description="Max Temperature")
    QV2M: float = Field(default=0.0, description="Specific Humidity")
    WS10M_RANGE: float = Field(default=0.0, description="Wind Speed Range at 10m")
    NVG_LAND: float = Field(default=0.0, description="Barren/Sparsely Vegetated Land")
    CULTRF_LAND: float = Field(default=0.0, description="Rainfed Cropland")
    CULTIR_LAND: float = Field(default=0.0, description="Irrigated Cropland")
    lat: float = Field(default=0.0, description="Latitude")
    lon: float = Field(default=0.0, description="Longitude")
    GRS_LAND: float = Field(default=0.0, description="Grassland")
    FOR_LAND: float = Field(default=0.0, description="Forest")
    CULT_LAND: float = Field(default=0.0, description="Total Cropland")
    T2MDEW: float = Field(default=0.0, description="Dew Point")
    T2MWET: float = Field(default=0.0, description="Wet Bulb Temperature")
    T2M_RANGE: float = Field(default=0.0, description="Temperature Range")
    TS: float = Field(default=0.0, description="Earth Skin Temperature")
    WS50M_RANGE: float = Field(default=0.0, description="Wind Speed Range at 50m")

class TimeSeriesPayload(BaseModel):
    data: list[DailyMetrics] = Field(
        ...,
        description="Chronological array of exactly 7 continuous days.",
        examples=[
            [
                {"PS":100.51,"elevation":63,"T2M_MAX":20.96,"QV2M":9.65,"WS10M_RANGE":1.46,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":13.51,"T2MWET":13.51,"T2M_RANGE":9.5,"TS":14.65,"WS50M_RANGE":2.81},
                {"PS":100.55,"elevation":63,"T2M_MAX":22.80,"QV2M":10.42,"WS10M_RANGE":1.60,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":14.71,"T2MWET":14.71,"T2M_RANGE":10.18,"TS":16.60,"WS50M_RANGE":2.41},
                {"PS":100.15,"elevation":63,"T2M_MAX":22.73,"QV2M":11.76,"WS10M_RANGE":2.67,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":16.52,"T2MWET":16.52,"T2M_RANGE":7.41,"TS":18.41,"WS50M_RANGE":3.66},
                {"PS":100.29,"elevation":63,"T2M_MAX":18.09,"QV2M":6.42,"WS10M_RANGE":3.59,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":6.09,"T2MWET":6.10,"T2M_RANGE":15.92,"TS":11.31,"WS50M_RANGE":5.58},
                {"PS":101.15,"elevation":63,"T2M_MAX":10.82,"QV2M":2.95,"WS10M_RANGE":1.98,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":-3.29,"T2MWET":-3.20,"T2M_RANGE":13.48,"TS":2.65,"WS50M_RANGE":4.19},
                {"PS":100.80,"elevation":63,"T2M_MAX":15.50,"QV2M":5.10,"WS10M_RANGE":2.10,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":4.00,"T2MWET":4.20,"T2M_RANGE":11.00,"TS":8.50,"WS50M_RANGE":3.20},
                {"PS":100.90,"elevation":63,"T2M_MAX":17.20,"QV2M":7.30,"WS10M_RANGE":1.80,"NVG_LAND":27.94,"CULTRF_LAND":56.29,"CULTIR_LAND":1.01,"lat":32.54,"lon":-86.64,"GRS_LAND":2.75,"FOR_LAND":10.71,"CULT_LAND":57.31,"T2MDEW":8.50,"T2MWET":8.80,"T2M_RANGE":9.20,"TS":12.10,"WS50M_RANGE":2.90}
            ]
        ]
    )

# --- Endpoint Definition ---
@router.post("/predict")
async def predict_drought_score(payload: TimeSeriesPayload, request: Request):
    # 1. Enforce sequence boundaries
    if len(payload.data) != 7:
        raise HTTPException(status_code=400, detail="Matrix error: Must contain exactly 7 days of inputs.")
    
    # 2. Access the shared model instance from the global app state safely
    model = getattr(request.app.state, "model", None)
    scaler = getattr(request.app.state, "scaler", None)  # ← get scaler

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    try:
        # 3. Transform incoming Pydantic validation object to raw nested lists
        sequence_matrix = []
        for metrics in payload.data:
            sequence_matrix.append([
                metrics.PS, metrics.elevation, metrics.T2M_MAX, metrics.QV2M, metrics.WS10M_RANGE,
                metrics.NVG_LAND, metrics.CULTRF_LAND, metrics.CULTIR_LAND, metrics.lat, metrics.lon,
                metrics.GRS_LAND, metrics.FOR_LAND, metrics.CULT_LAND, metrics.T2MDEW, metrics.T2MWET,
                metrics.T2M_RANGE, metrics.TS, metrics.WS50M_RANGE
            ])
            
        # Scale each day's features before passing to model
        raw = np.array(sequence_matrix, dtype=np.float32)       # shape (7, 18)
        scaled = (raw - scaler["mean"]) / scaler["std"]                           # ← scale here
        
        # 4. Format array exactly for PyTorch input shape expectations
        input_tensor = torch.tensor([scaled], dtype=torch.float32)  # shape (1, 7, 18)
        
        # 5. Execute Deep Learning Inference
        with torch.no_grad():
            prediction = model(input_tensor)
            
        return {
            "status": "success",
            "predicted_drought_score": round(prediction.item(), 2)
        }
        
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(err)}")