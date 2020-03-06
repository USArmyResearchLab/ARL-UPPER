from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from .upper import UpperFingerprint
from .other_fps import OtherFingerprint
import logging
import numpy as np
import sys
import os
import re

sys.path.append(os.getcwd())
from params import (
    fp_params,
    data_params,
    split_params,
    xgb_train_params,
    save_load_params,
)

sys.path.remove(os.getcwd())

Xy_type = Tuple[pd.core.frame.DataFrame, pd.core.series.Series]
train_test_data_type = Tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.series.Series,
    pd.core.series.Series,
]


def construct_fingerprint(
    data_path: str,
    smiles_name: str,
    labels: dict,
    fp_type: str,
    fp_path: str,
    d_path: str,
) -> None:
    """Load in smiles, construct and save fingerprint."""

    # load dataset
    g = pd.read_csv(data_path)

    start = time.time()
    fp = (
        UpperFingerprint(g[smiles_name], labels)
        if fp_type == "upper"
        else OtherFingerprint(g[smiles_name], fp_type)
    )
    end = time.time()

    # log time
    logging.info(
        "{} fingerprint for {} molecules: {} s".format(fp_type, g.shape[0], end - start)
    )

    # save fingerprint
    fp.output_to_csv(fp_path)

    # save data array
    fp.output_darray(d_path)


def check_string_to_float(s: str) -> bool:
    """Check if string can be converted to float."""

    try:
        float(s)
        return True
    except:
        return False


def nonfloatable_indices(y: np.ndarray) -> list:
    """Find bad indices that cannot be converted to float."""

    return [i for (i, x) in enumerate(y) if not check_string_to_float(x)]


def clean_target(f: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Remove rows with target values that cannot be converted to float or nan."""

    # experimental value (y)
    y = f.iloc[:, -1]

    # indices that cannot be converted to float
    nonfloatable_indices_ = nonfloatable_indices(y.values)

    # drop bad indices
    f = f.drop(f.index[nonfloatable_indices_], axis=0)

    # remove nan
    return f.dropna(axis=0)


def prepare_data(fp_path: str, target_path: str, target_name: str) -> Xy_type:
    """Prepare (X: fingerprint, y: target) combo and remove rows with missing/nan targets."""

    # load fingerprint
    f = pd.read_csv(fp_path)

    # experimental properties
    g = pd.read_csv(target_path)

    # add experimental value to fingerprint dataframe
    f["target"] = g[target_name]

    # remove rows with nonfloatable and nan
    f = clean_target(f)

    # fingerprint (X) and experimental value (y)
    X, y = f.iloc[:, :-1], f.iloc[:, -1]

    # replace [,],< with _ in column names
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in X.columns.values
    ]

    return X, y


def prepare_train_test(
    X: pd.core.frame.DataFrame,
    y: pd.core.series.Series,
    p: Optional[int],
    split: dict = {"train": 1.0, "test": 0.0},
    tt_indices_path: str = "",
    train_indices: np.ndarray = np.array([]),
    test_indices: np.ndarray = np.array([]),
    random_state: Optional[int] = None,
) -> train_test_data_type:
    """Train/test data."""

    # train/test indices identified in npz file
    if tt_indices_path:
        indices = np.load(tt_indices_path, allow_pickle=True)
        train_indices, test_indices = indices["train"][p], indices["test"][p]

        X_train, X_test, y_train, y_test = (
            X.iloc[train_indices],
            X.iloc[test_indices],
            y.iloc[train_indices],
            y.iloc[test_indices],
        )

    # train/test indices defined for cv
    elif train_indices.size != 0 or test_indices.size != 0:

        X_train, X_test, y_train, y_test = (
            X.iloc[train_indices],
            X.iloc[test_indices],
            y.iloc[train_indices],
            y.iloc[test_indices],
        )

    # train to all
    elif split["train"] == 1.0 or split["test"] == 0.0:

        X_train, X_test, y_train, y_test = X, X, y, y

    # train/test random split
    else:
        random_state = random_state if random_state else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split["test"], random_state=random_state
        )

    return X_train, X_test, y_train.astype(float), y_test.astype(float)


def log_inputs() -> None:
    """Log input parameters."""

    logging.info("fp_params: {}".format(fp_params))
    logging.info("data_params: {}".format(data_params))
    logging.info("split_params: {}".format(split_params))
    logging.info("xgb_train_params: {}".format(xgb_train_params))
    logging.info("save_load_params: {}".format(save_load_params))
