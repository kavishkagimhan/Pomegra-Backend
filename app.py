from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import pandas as pd
import os
import pickle


current_directory = os.getcwd()
# Load your machine learning models
aphid_model = joblib.load("./models/climate/aphid_model.pkl")
whiteflies_model = joblib.load("./models/climate/whiteflies_model.pickle")
antraconos_model = joblib.load("./models/climate/antraconos_model.pickle")
bacterial_blight = joblib.load("./models/climate/bacterial_blight.pickle")

file_path = os.path.join(current_directory, './models/soil/Fertilizer.csv')
FERTILIZER_DATA = pd.read_csv(file_path)

model_path = os.path.join(current_directory, './models/soil/svm_model.pkl')

# Create a FastAPI app
app = FastAPI()
with open(model_path, 'rb') as model_file:
    MODEL = pickle.load(model_file)
# Define input and output data models
class InputData(BaseModel):
    temperature: float
    humidity: float

class OutputData(BaseModel):
    status: str  # Change the output field to a string

class FertilizerInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float



class FertilizerOutput(BaseModel):
    suggested_fertilizer: str



# Function to map status_num to descriptive labels
def get_status_label(status_num):
    if status_num == 1:
        return "high"
    elif status_num == 2:
        return "medium"
    elif status_num == 3:
        return "low"
    else:
        return "unknown"

# Create a prediction endpoint for aphid model
@app.post("/predict_aphid", response_model=OutputData)
async def predict_aphid(input_data: InputData):
    input_values = np.array([input_data.temperature, input_data.humidity]).reshape(1, -1)
    prediction = aphid_model.predict(input_values)
    status_label = get_status_label(int(prediction[0]))
    return {"status": status_label}

# Create a prediction endpoint for whiteflies model
@app.post("/predict_whiteflies", response_model=OutputData)
async def predict_whiteflies(input_data: InputData):
    input_values = np.array([input_data.temperature, input_data.humidity]).reshape(1, -1)
    prediction = whiteflies_model.predict(input_values)
    status_label = get_status_label(int(prediction[0]))
    return {"status": status_label}

# Create a prediction endpoint for antraconos model
@app.post("/predict_antraconos", response_model=OutputData)
async def predict_antraconos(input_data: InputData):
    input_values = np.array([input_data.temperature, input_data.humidity]).reshape(1, -1)
    prediction = antraconos_model.predict(input_values)
    status_label = get_status_label(int(prediction[0]))
    return {"status": status_label}

# Create a prediction endpoint for bacterial blight model
@app.post("/predict_bacterial_blight", response_model=OutputData)
async def predict_bacterial_blight(input_data: InputData):
    input_values = np.array([input_data.temperature, input_data.humidity]).reshape(1, -1)
    prediction = bacterial_blight.predict(input_values)
    status_label = get_status_label(int(prediction[0]))
    return {"status": status_label}


@app.post("/suggest-fertilizers", response_model=FertilizerOutput)
async def suggest_fertilizers(input_data: FertilizerInput):
    nitrogen = input_data.nitrogen
    phosphorus = input_data.phosphorus
    potassium = input_data.potassium

    suggested_fertilizer = get_suggested_fertilizers(nitrogen, phosphorus, potassium)
    return FertilizerOutput(suggested_fertilizer=suggested_fertilizer)

def get_suggested_fertilizers(nitrogen, phosphorus, potassium):
    min_distance = float('inf')
    suggested_fertilizer = None

    for _, row in FERTILIZER_DATA.iterrows():
        distance = (
            (row['Nitrogen'] - nitrogen) ** 2 +
            (row['Phosphorous'] - phosphorus) ** 2 +
            (row['Potassium'] - potassium) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            suggested_fertilizer = row['Fertilizer Name']

    return suggested_fertilizer if suggested_fertilizer else "No suitable fertilizer found"




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8005)
