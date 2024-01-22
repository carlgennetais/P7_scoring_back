"""
Load data from disk.
"""

import os
import pickle

import pandas as pd


class DataLoader:
    """Load data from disk."""

    data = "./data/"
    models = "./models/"

    @classmethod
    def df(
        cls,
        fn: str = "app_train_cleaned.pkl",
    ):
        """Load data from disk."""

        # filename
        _fn = os.path.join(cls.data, fn)

        # read df
        customers = pd.read_pickle(_fn)

        # udate df
        customers.set_index("SK_ID_CURR").drop("TARGET", axis=1)

        return customers

    @classmethod
    def model(
        cls,
        fn: str = "model.pkl",
    ):
        """Load model from disk."""

        # filename
        _fn = os.path.join(cls.models, fn)

        # load model
        with open(_fn, "rb") as f:
            model = pickle.load(f)

        return model

    @classmethod
    def shap(cls):
        """Load shap values from disk."""

        # # filename
        # _fn = os.path.join(cls.models, "shap_values.pkl")

        # # load model
        # with open(_fn, "rb") as f:
        #     shap_values = pickle.load(f)

        # return shap_values
