import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("bin_model.pkl")
vectorizer = joblib.load("bin_vectorizer.pkl")
waste_types = pd.read_csv("waste_types.csv")


@app.get("/")
def home():
    return {"message": "Bin tool is running"}


@app.get("/classify")
def classify(item: str):
    new_X = vectorizer.transform([item])

    if new_X.nnz == 0:
        return {"answer": "Not sure - ask a staff member"}

    category = model.predict(new_X)[0]
    row = waste_types[waste_types["Waste_Type"] == category].iloc[0]
    confidence = model.predict_proba(new_X).max()

    return {
        "item": item,
        "category": str(category),
        "bin_color": str(row["Bin_Color"]),
        "hazardous": int(row["Hazardous"]),
        "confidence": round(float(confidence), 2),
    }