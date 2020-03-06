import sys
import os

sys.path.append(os.getcwd())
from params import (
    xgb_train_params,
    save_load_params,
    xgb_hyper_params,
    xgb_train_params_final,
    split_params,
    multiprocess_params,
    lr_train_params,
    lr_hyper_params,
    procedure_params,
)

sys.path.remove(os.getcwd())
from .utils import prepare_train_test
from .ml import xgb_model, lr_model
from typing import Tuple, Callable, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import multiprocessing as mp
import functools

train_test_data_type = Tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.series.Series,
    pd.core.series.Series,
]

model_procedure = dict()


def register(func: Callable) -> Callable:
    """Register a function."""

    model_procedure[func.__name__] = func
    return func


@register
def xgb(train_test_data: train_test_data_type) -> None:
    """Execute xgb procedure."""

    # initialize model
    model = xgb_model(train_test_data)

    # train
    if procedure_params["train"]:
        model.train(xgb_train_params, save_load_params)

    # hyperparametrize (and train)
    if procedure_params["hyperparametrize"]:
        model.train(xgb_train_params, save_load_params)
        model.hypersearch(xgb_hyper_params)
        model.train(xgb_train_params_final, save_load_params)

    # feature reduction
    if procedure_params["feature_reduction"]:
        model.feature_reduction(save_load_params)

    # predict
    if procedure_params["predict"]:
        model.predict(save_load_params)

    # save model
    model.save_model(save_load_params)


@register
def lr(train_test_data: train_test_data_type) -> None:
    """Execute lr procedure."""

    # initialize model
    model = lr_model(train_test_data)

    # train model
    if procedure_params["train"]:
        model.train(lr_train_params, save_load_params)

    # hyperparameter search
    if procedure_params["hyperparametrize"]:
        model.train(lr_train_params, save_load_params)
        model.hypersearch(lr_hyper_params)
        model.train(sl_params=save_load_params)

    # predict
    if procedure_params["predict"]:
        model.predict(save_load_params)

    # save model
    model.save_model(save_load_params)


def run_single_model(
    X: pd.core.frame.DataFrame, y: pd.core.series.Series, p: Optional[int] = None
) -> None:
    """Run a single ML model."""

    # split train, test
    train_test_data = prepare_train_test(X, y, p, **split_params)

    # run model
    model_procedure[procedure_params["model"]](train_test_data)


def single_model_mp(
    indices: Tuple[np.ndarray, np.ndarray],
    X: pd.core.frame.DataFrame,
    y: pd.core.series.Series,
) -> None:
    """Run single ML model."""

    # unpack train, test indices
    train_indices, test_indices = indices

    # rewrite split_params
    split_params["train_indices"] = train_indices
    split_params["test_indices"] = test_indices

    # single model
    run_single_model(X, y)


def run_multiprocessing(
    single_model: Callable, X: pd.core.frame.DataFrame, y: pd.core.series.Series
) -> None:
    """Run ML model across processors."""

    # k-fold cv
    cv = KFold(**multiprocess_params)

    # train, test indices for cv
    args = list(cv.split(X, y))

    # save train, test indices
    # train faster by running each model on all cores of node on hpc system
    # unpacked_args = list(zip(*args))
    # np.savez('train-test_indices', train = unpacked_args[0], test = unpacked_args[1])
    # exit()

    # multiprocessing: execute code with different splits
    with mp.Pool() as pool:  # use all available cores
        pool.map(functools.partial(single_model, X=X, y=y), args)
