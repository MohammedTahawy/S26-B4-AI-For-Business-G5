from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Iris Species Prediction API",
    description="API to predict Iris flower species using the pre-trained model.",
    version="1.0.0"
)

# Load target names for mapping prediction index to name
iris_data = load_iris()
target_names = iris_data.target_names  # ['setosa', 'versicolor', 'virginica']

# Load model
try:
    model = joblib.load("iris_model.pkl")
except Exception as e:
    print(f"Error loading model with joblib: {e}")
    model = None

# Define input data schema using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API. Use /docs to view Swagger documentation."}

@app.post("/predict")
def predict_species(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file iris_model.pkl could not be loaded.")
    
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    prediction_idx = int(model.predict(input_data)[0])
    predicted_species = str(target_names[prediction_idx])
    
    return {
        "prediction_class": prediction_idx,
        "predicted_species": predicted_species
    }

if __name__ == "__main__":
    uvicorn.run("iris_api:app", host="0.0.0.0", port=8000, reload=True)
