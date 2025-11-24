from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dutch_gp_predictor import DutchGPPredictor
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F1-AI-Predictor Dutch GP API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-vercel-app.vercel.app"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for prediction response
class Prediction(BaseModel):
    position: int
    driver_id: str
    constructor_id: str
    grid: int
    win_probability: float
    podium_probability: float
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

# Initialize predictor
try:
    predictor = DutchGPPredictor(
        model_dir="E:/Shivam/F1/f1-ai-predictor/models",
        data_path="E:/Shivam/F1/f1-ai-predictor/data/f1_race_data.csv",
        weather_api_key="87e5ba753df3d9e4f6fc96f167f6a7a6"
    )
    if not predictor.load_model_and_data():
        logger.error("Failed to initialize predictor")
        raise Exception("Predictor initialization failed")
except Exception as e:
    logger.error(f"Error initializing predictor: {str(e)}")
    raise Exception(f"Predictor initialization failed: {str(e)}")

@app.get("/predict", response_model=PredictionResponse)
async def predict_dutch_gp():
    """Get 2025 Dutch GP predictions"""
    try:
        logger.info("Generating predictions for 2025 Dutch GP...")
        predictions_df = predictor.predict_dutch_gp()
        
        if predictions_df is None or predictions_df.empty:
            logger.error("No predictions returned")
            raise HTTPException(status_code=500, detail="Prediction failed: No results returned")
        
        # Convert DataFrame to list of Prediction objects
        predictions = [
            Prediction(
                position=index + 1,
                driver_id=row["driver_id"],
                constructor_id=row["constructor_id"],
                grid=int(row["grid"]),
                win_probability=float(row["win_probability"] * 100),  # Convert to %
                podium_probability=float(row["podium_probability"] * 100),  # Convert to %
                confidence=float(row["confidence"] * 100)  # Convert to %
            )
            for index, row in predictions_df.iterrows()
        ]
        
        # Save predictions and visualization
        predictor.save_predictions(predictions_df)
        predictor.create_visualizations(predictions_df)
        
        logger.info("Predictions generated successfully")
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "circuit": "Dutch GP"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port to avoid conflicts
