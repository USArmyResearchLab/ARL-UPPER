from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from .utils import reformat_X_test, mae, rmse
from typing import Union
import pandas as pd
import numpy as np
from multiprocessing import current_process
import logging
import pickle
import xgboost as xgb
import sklearn
from sklearn.linear_model import Ridge

model_types = Union[xgb.sklearn.XGBRegressor, sklearn.linear_model.ridge.Ridge]


def train_subroutine(
    sl_params: dict,
    model: model_types,
    X_train: pd.core.frame.DataFrame,
    y_train: pd.core.series.Series,
) -> None:
    """Train model."""

    # train
    model.fit(X_train, y_train)

    # predict on train set
    preds_train = model.predict(X_train)

    # output to npz
    np.savez(
        "{}-{}".format(sl_params["npz_train_path"], current_process().name),
        indices=y_train.index,
        true=y_train,
        preds=preds_train,
    )

    # log train performance
    logging.info(
        "train: {} (mae), {} (rmse)".format(
            mae(y_train, preds_train), rmse(y_train, preds_train)
        )
    )


def hypersearch_subroutine(
    hsearch_params: list,
    model: model_types,
    X_train: pd.core.frame.DataFrame,
    y_train: pd.core.series.Series,
) -> None:
    """Sequential hyperparameter search."""

    for hsearch_param in hsearch_params:

        # grid
        grid = GridSearchCV(
            estimator=model, param_grid=hsearch_param, cv=10, n_jobs=-1, verbose=3
        )

        # train
        grid.fit(X_train, y_train)

        # log params, best score, best params
        logging.info(
            "h-search param: {}, best score: {}, best params: {}".format(
                hsearch_param, grid.best_score_, grid.best_params_
            )
        )

        # redefine parameters of model
        model.set_params(**grid.best_params_)


def predict_subroutine(
    sl_params: dict,
    model: model_types,
    X_test: pd.core.frame.DataFrame,
    y_test: pd.core.series.Series,
) -> None:
    """Predict on test set."""

    # model or load model
    model = (
        pickle.load(open(sl_params["load_model_path"], "rb"))
        if sl_params["load_model"]
        else model
    )

    # test set fp, reformated to training fp
    X_test = reformat_X_test(sl_params, X_test)

    # predict on test set
    preds_test = model.predict(X_test)

    # output to npz
    np.savez(
        "{}-{}".format(sl_params["npz_test_path"], current_process().name),
        indices=y_test.index,
        true=y_test,
        preds=preds_test,
    )

    # log test performance
    logging.info(
        "test: {} (mae), {} (rmse)".format(
            mae(y_test, preds_test), rmse(y_test, preds_test)
        )
    )


def feature_reduction_subroutine(
    sl_params: dict,
    model: model_types,
    X_train: pd.core.frame.DataFrame,
    y_train: pd.core.series.Series,
    X_test: pd.core.frame.DataFrame,
    y_test: pd.core.series.Series,
) -> None:
    """Reduce trained model by feature importances of xgb model."""

    # model or load model
    model = (
        pickle.load(open(sl_params["load_model_path"], "rb"))
        if sl_params["load_model"]
        else model
    )

    # params
    xgb_params = model.get_xgb_params()

    # feature importance
    thresholds = np.unique(model.feature_importances_)

    # initialize features names, reduction stats
    feat_names = [""] * len(thresholds)
    feat_red = np.zeros((len(thresholds), 3))

    for (i, thresh) in enumerate(thresholds):

        # select features from threshold
        select = SelectFromModel(model, threshold=thresh, prefit=True)

        # select features
        feature_idx = select.get_support()
        feat_names[i] = X_train.columns[feature_idx]

        # reduce features
        (select_X_train, select_X_test) = (
            select.transform(X_train),
            select.transform(X_test),
        )

        # train
        selection_model = xgb.XGBRegressor(**xgb_params)
        selection_model.fit(select_X_train, y_train)

        # predict on test set
        preds_test = selection_model.predict(select_X_test)

        # (n, rmse, mae) and feature names
        feat_red[i] = [
            select_X_train.shape[1],
            mae(y_test, preds_test),
            rmse(y_test, preds_test),
        ]

    # log performances
    logging.info(
        "feature reduction test performance (n, mae, rmse): {}".format(feat_red)
    )

    # log best performance and names
    best_rmse_index = np.argmin(feat_red[:, 2])
    logging.info(
        "feature reduction best performance (n, mae, rmse): {}, best names: {}".format(
            feat_red[best_rmse_index], feat_names[best_rmse_index]
        )
    )


def save_model_subroutine(sl_params: dict, model: model_types) -> None:
    """Save model."""

    if model and sl_params["save_model"]:
        pickle.dump(
            model,
            open(
                "{}-{}.pkl".format(
                    sl_params["save_model_path"], current_process().name
                ),
                "wb",
            ),
        )
