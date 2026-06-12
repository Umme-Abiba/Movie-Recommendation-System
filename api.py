from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd


app = FastAPI()

model = joblib.load(
    "artifacts/model/model.pkl"
)


class MovieData(BaseModel):

    budget: float

    genres: int

    popularity: float

    runtime: float

    vote_average: float

    release_year: int


@app.get("/")
def home():

    return {
        "message": "Movie Revenue Prediction API Running"
    }


@app.post("/predict")
def predict(data: MovieData):

    input_data = pd.DataFrame(
        [{
            "budget": data.budget,
            "genres": data.genres,
            "popularity": data.popularity,
            "runtime": data.runtime,
            "vote_average": data.vote_average,
            "release_year": data.release_year
        }]
    )

    prediction = model.predict(input_data)

    return {
        "predicted_revenue": float(prediction[0])
    }