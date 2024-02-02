"""
Main API code
"""


# builtin
import pickle
from typing import Union

# libraires
import pandas as pd
from fastapi import FastAPI, HTTPException

# source code
from .loaders import DataLoader

app = FastAPI()

# load cleaned-transformed dataframe once
# use raw data instead for interpretability ?


data = DataLoader.df()
# load model and shap explainer
with open("../../models/model.pkl", "rb") as f:
    model = pickle.load(f)
f.close()
with open("../../models/shap_explainer.pkl", "rb") as f:
    shap_explainer = pickle.load(f)
f.close()


# def get_customer(customer_id: id):
#     if customer_id in customers.index:
#         return customers.loc[customer_id, :]
#     else:
#         return -1


# API routes
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ping")
def ping():
    return {"Ping successfull"}


@app.get("/customers")
def list_customers():
    """ADD A DOCSTRING HERE"""

    # TODO performance
    return data.head(100).index.to_list()


@app.get("/customers/{customer_id}")
def read_single_customer(customer_id: int):
    """ADD A DOCSTRING HERE"""

    if customer_id not in data.index:
        raise HTTPException(status_code=404, detail="Customer ID does not exist")

    return data.loc[customer_id, :].fillna("").to_dict()


@app.get("/customers_stats")
def all_customers_stats():
    """ADD A DOCSTRING HERE"""

    return data.describe().loc[["count", "mean", "std"], :].to_dict()


# @app.get("/customers/{customer_id}/predict")
@app.get("/predict/{customer_id}")
def predict(customer_id: int, q: Union[str, None] = None):
    """ADD A DOCSTRING HERE"""

    if customer_id not in data.index:
        raise HTTPException(status_code=404, detail="Customer ID does not exist")

    model = DataLoader.model()

    pred = model.predict_proba(pd.DataFrame(data.loc[customer_id, :]).T)

    return {"prediction": pred}


@app.get("/shap/{customer_id}")
def shap_values(customer_id: int, q: Union[str, None] = None):
    return {}
