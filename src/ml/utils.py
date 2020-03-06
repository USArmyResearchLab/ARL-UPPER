from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import sklearn
from sklearn.linear_model import Ridge
import re

model_types = Union[xgb.sklearn.XGBRegressor, sklearn.linear_model.ridge.Ridge]


def UseCV(
    model: xgb.sklearn.XGBRegressor,
    data: pd.core.frame.DataFrame,
    label: pd.core.series.Series,
    nfold: int = 10,
    metrics: list = ["mae", "rmse"],
    early_stopping_rounds: int = 50,
) -> None:
    """Determine n_estimators of model using cv."""

    # model params
    xgb_params = model.get_xgb_params()

    # train data
    xgb_train = xgb.DMatrix(data=data, label=label)

    # cross validation
    cv = xgb.cv(
        xgb_params,
        xgb_train,
        num_boost_round=xgb_params["n_estimators"],
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=early_stopping_rounds,
    )

    # log cv results
    logging.info("cv:\n{}".format(cv.tail()))

    # model with new n_estimators
    model.set_params(n_estimators=cv.shape[0])


def GetParams(model: model_types) -> dict:
    """Get model parameters."""

    # xgb params
    if model.__class__ == xgb.sklearn.XGBRegressor:
        params = model.get_xgb_params()

    # lr params
    if model.__class__ == sklearn.linear_model.ridge.Ridge:
        params = model.get_params()

    return params


def ReplaceParams(model: model_types, replace_params: dict) -> dict:
    """Replace model parameters."""

    # model params
    params = GetParams(model)

    # replace params
    for param in replace_params:
        params[param] = replace_params[param]

    return params


def Reformat(
    new: pd.core.frame.DataFrame, old: pd.core.frame.DataFrame, index: int
) -> None:
    """Reformat fingerprint of test compounds according to features of model."""

    new.at[index, old.columns] = old.loc[index, old.columns]


def reformat_X_test(
    sl_params: dict, X_test_old: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """X_test reformatted according to trained model."""

    # load features
    train_fp = pd.read_csv(sl_params["train_fp_path"])

    # zeros dataframe
    X_test = pd.DataFrame(0, index=X_test_old.index, columns=train_fp.columns)

    # replace [,],< with _ in column names
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X_test.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in X_test.columns.values
    ]

    # reformat fingerprint
    Reformat(X_test, X_test_old, X_test.index)

    return X_test


def rmse(true: pd.core.series.Series, preds: np.ndarray) -> float:
    """Root-mean-square-error."""

    return np.sqrt(mean_squared_error(true, preds))


def mae(true: pd.core.series.Series, preds: np.ndarray) -> float:
    """Mean-absolute-error."""

    return mean_absolute_error(true, preds)
