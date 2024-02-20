import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# load cleaned-transformed dataframe once
# use raw data instead for interpretability ?


customers = (
    pd.read_pickle("./data/processed/app_train_cleaned_sample.pkl")
    .set_index("SK_ID_CURR")
    .drop("TARGET", axis=1)
)

# load model and shap explainer
with open("./models/model.pkl", "rb") as f:
    model = pickle.load(f)
f.close()
with open("./models/shap_explainer.pkl", "rb") as f:
    shap_explainer = pickle.load(f)
f.close()


def get_customer(customer_id: int):
    """
    Return data for one customer
    """
    if customer_id not in customers.index:
        raise HTTPException(status_code=404, detail="Customer ID does not exist")
    return customers.loc[customer_id, :]


# API routes
@app.get("/")
def read_root():
    return {"Ping successfull"}


@app.get("/customers")
def list_customers():
    # TODO performance
    return customers.head(1000).index.to_list()


@app.get("/customers/{customer_id}")
def read_single_customer(customer_id: int):
    if customer_id not in customers.index:
        raise HTTPException(
            status_code=404, detail="Customer ID is invalid or does not exist"
        )
    return customers.loc[customer_id, :].fillna("").to_dict()


@app.get("/customers_stats")
def all_customers_stats():
    return customers.describe().loc[["count", "mean", "std"], :].T.to_dict()


# @app.get("/customers/{customer_id}/predict")
@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    if customer_id not in customers.index:
        raise HTTPException(status_code=404, detail="Customer ID does not exist")
    probaList = model.predict_proba(pd.DataFrame(customers.loc[customer_id, :]).T)[0]
    return {0: probaList[0], 1: probaList[1]}


@app.get("/shap/{customer_id}")
def shap_values(customer_id: int):
    """
    Return 20
    TODO
    """
    if customer_id not in customers.index:
        raise HTTPException(status_code=404, detail="Customer ID does not exist")
    shap_for_sample = pd.DataFrame(
        shap_explainer.shap_values(customers.loc[customer_id, :])
    ).fillna(0)
    shap_scaled = StandardScaler().fit_transform(shap_for_sample)
    shap_scaled = pd.DataFrame(shap_scaled, index=customers.columns)
    shap_scaled = pd.Series(shap_scaled.iloc[:, 0])
    top_shap = shap_scaled.sort_values(ascending=False).head(10)
    bottom_shap = shap_scaled.sort_values(ascending=True).head(10)
    return {"top": top_shap.to_dict(), "bottom": bottom_shap.to_dict()}
