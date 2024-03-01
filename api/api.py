import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# load cleaned-transformed dataframe once
# TODO: use raw data instead for interpretability ?
customers = (
    pd.read_pickle("./data/processed/app_train_cleaned_sample.pkl")
    .set_index("SK_ID_CURR")
    .drop("TARGET", axis=1)
)

# load model and shap explainer
with open("./models/model.pkl", "rb") as f:
    model = pickle.load(f)
f.close()
with open("./models/shap_explainer.pkl", "rb") as f2:
    shap_explainer = pickle.load(f2)
f2.close()


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
    Predict probability for a selected customer of repaying a loan application.

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    dict
        0: probability of class 0
        1: probability of class 1
    """
    probaList = model.predict_proba(pd.DataFrame(get_customer(customer_id)).T)[0]
    return {0: probaList[0], 1: probaList[1]}


@app.get("/shap/{customer_id}")
def shap_values(customer_id: int):
    """
    Get shap_values for local

    Parameters
    ----------
    customer_id : int
        Index of selected customer. If index is out of range, http error 404 is raised.

    Returns
    -------
    dict
        Nested dict:
        Level 0: 'top' for
    """
    # TODO : instead of 10 top and 10 bottom, 20 most contributing?
    shap_for_sample = pd.DataFrame(
        shap_explainer.shap_values(get_customer(customer_id))
    ).fillna(0)
    shap_scaled = StandardScaler().fit_transform(shap_for_sample)
    shap_scaled = pd.DataFrame(shap_scaled, index=customers.columns)
    shap_scaled = pd.Series(shap_scaled.iloc[:, 0])
    top_shap = shap_scaled.sort_values(ascending=False).head(10)
    bottom_shap = shap_scaled.sort_values(ascending=True).head(10)
    return {"top": top_shap.to_dict(), "bottom": bottom_shap.to_dict()}
