from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Define input data schema
class PatientData(BaseModel):
    age: float
    sex: float  
    bmi: float
    bp: float   # blood pressure
    s1: float   # serum measurement 1
    s2: float   # serum measurement 2  
    s3: float   # serum measurement 3
    s4: float   # serum measurement 4
    s5: float   # serum measurement 5
    s6: float   # serum measurement 6
    
    class Config:
        schema_extra = {
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.06,
                "bp": 0.02,
                "s1": -0.04,
                "s2": -0.04,
                "s3": -0.02,
                "s4": -0.01,
                "s5": 0.01,
                "s6": 0.02
            }
        }

# initialize FastAPI app
app = FastAPI(
    title="Diabetes Progression Predictor",
    description="Predicts diabetes progression score from physiological features",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load the trained model
model_path = os.path.join("models", "diabetes_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def get_interpretation(score):
    if score < 100:
        return "Low risk of diabetes progression"
    elif score < 200:
        return "Moderate risk of diabetes progression"
    else:
        return "High risk of diabetes progression"

@app.get('/')
def read_root():
    return FileResponse('app/templates/index.html')

@app.post('/predict')
def predict_progression(patient: PatientData):
    """
    Predict diabetes progression score
    """
    features = np.array([[
        patient.age, patient.sex, patient.bmi, patient.bp,
        patient.s1, patient.s2, patient.s3, patient.s4,
        patient.s5, patient.s6
    ]])

    prediction = model.predict(features)[0]
    return {
        'predicted_progression_score': round(prediction, 2),
        'interpretation': get_interpretation(prediction)
    }

@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'model': 'diabetes_progression_v1'
    }