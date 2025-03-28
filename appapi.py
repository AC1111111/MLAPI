from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


class transactionItem(BaseModel):
    HomeFoodExpenditure: float
    RestaurantAndHotelExpenditure: float
    ClothingExpenditure: float
    HousingExpenditure: float
    Rent: float
    PublicTransportationExpenditure: float
    CommExpenditure: float
    MiscelleneousExpenditure: float
    BusinessCash: float
    PersonalVehicleExpenditure: float

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"],
)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:transactionItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    prediction = model.predict(df)
    return {"prediction": int(prediction)}