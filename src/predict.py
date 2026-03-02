import pickle
import numpy as np

def predict(input_data):
    model = pickle.load(open("models/model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0][1]