import pickle

import bz2file as bz2
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

# app = FastAPI()
# FastAPI with non-default Json that allows np.nan
app = FastAPI(default_response_class=ORJSONResponse)

# load cleaned-transformed dataframe once
# TODO: use raw data instead for interpretability ?
customers = pd.read_pickle("./data/processed/data_cleaned_sample.pkl").drop(
    ["TARGET", "index"], axis=1
)
customers.set_index("SK_ID_CURR", inplace=True, drop=True)

# load model and shap explainer
with bz2.BZ2File("./models/model.pbz2", "rb") as f:
    model = pickle.load(f)
f.close()
with open("./models/shap_explanation.pkl", "rb") as f2:
    exp = pickle.load(f2)
f2.close()
# load best threshold for this model
PROBA_THRESHOLD = pd.read_csv("./models/model_proba_threshold.csv").value[0]


def get_customer(customer_id: int):
    """
    Get data of a single customer, selected by id. All columns are included.
    Internal function called in multiple API routes.

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    Series
        Series containing all data about customer, one feature per row.
    """
    if customer_id not in customers.index:
        raise HTTPException(status_code=404, detail="Customer ID does not exist")
    return customers.loc[customer_id, :]


# shap explainer is a special data structure contains 3 arrays (values, base_values and data)
# to send it via the API we need to encode it in json and back :
# exp -> json -> exp
def explanation_to_dict(single_exp: shap._explanation.Explanation) -> dict:
    """
    Convert Shap explanation to dict
    """
    v = pd.DataFrame(single_exp.values)[0].to_dict()
    d = pd.DataFrame(single_exp.data)[0].to_dict()
    dd = single_exp.display_data.to_dict()
    b = single_exp.base_values
    return {"values": v, "base_values": b, "data": d, "display_data": dd}


# API routes
@app.get("/")
def read_root():
    """
    Display a ping successfull message, for debugging purposes.

    Returns
    -------
    String
        "Ping successfull"

    """
    return {"Ping successfull"}


@app.get("/customers")
def list_customers():
    """
    Lists all existing customers by id. Limited to the first 1000 rows because of free online hosting conditions.

    Returns
    -------
    List
        The ids of all customers.
    """
    return customers.head(1000).index.to_list()


@app.get("/customers/{customer_id}")
def read_single_customer(customer_id: int):
    """
    Get dict of data of a single customer, selected by id. All features are included.

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    dict
        dict containing all data about customer, one feature per key.
        Missing values are shown as empty string.
    """
    return get_customer(customer_id).fillna("").to_dict()


@app.get("/customers_stats")
def all_customers_stats():
    """
    Get basic statistics (count, mean and standard deviation) of all customer population, for each numerical feature.

    Returns
    -------
    dict
        Nested dict.
        Level 0: measure (count/mean/std)
        Level 1: feature names and values
    """
    return customers.describe().loc[["count", "mean", "std"], :].T.to_dict()


@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    """
    Predict ability for a selected customer of repaying a loan application.

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    int:
        predicted class (0 or 1)
    """
    proba = model.predict_proba(pd.DataFrame(get_customer(customer_id)).T)[0][1]
    if proba > PROBA_THRESHOLD:
        return {"loan_result": 1}
    else:
        return {"loan_result": 0}


@app.get("/shap/{customer_id}")
def shap_values(customer_id: int):
    """
    Get the shap values for a selected customer.
    These features are have the most impact on model prediction for this specific customer (local explainer)

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    shap._explanation.Explanation
        Shap explanation object for a specific customer

    """
    # TODO:docstring
    # check customer_id is valid
    _ = get_customer(customer_id)
    idx = customers.index.get_loc(customer_id)
    return explanation_to_dict(exp[idx])
