from typing import Tuple, Optional
import xgboost as xgb
from .utils import UseCV, ReplaceParams
from .subroutine import (
    train_subroutine,
    hypersearch_subroutine,
    predict_subroutine,
    feature_reduction_subroutine,
    save_model_subroutine,
)
import logging
import pandas as pd
from sklearn.linear_model import Ridge

train_test_data_type = Tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.series.Series,
    pd.core.series.Series,
]


class xgb_model(object):
    """xgb model."""

    def __init__(self, train_test_data: train_test_data_type) -> None:
        """Initialize class attributes."""

        # model
        self.model: Optional[xgb.sklearn.XGBRegressor] = None

        # train, test subsets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_data

    def train(
        self, xgb_params: dict = {}, sl_params: dict = {}, use_cv: bool = True
    ) -> None:
        """Train xgb model."""

        # replace params
        xgb_params = (
            ReplaceParams(self.model, replace_params=xgb_params)
            if self.model
            else xgb_params
        )

        # initiate model
        self.model = xgb.XGBRegressor(**xgb_params)

        # use cv to determine n_estimators
        if use_cv:
            UseCV(model=self.model, data=self.X_train, label=self.y_train)

        # train
        train_subroutine(sl_params, self.model, self.X_train, self.y_train)

    def predict(self, sl_params: dict = {}) -> None:
        """Predict with trained xgb model."""

        predict_subroutine(sl_params, self.model, self.X_test, self.y_test)

    def hypersearch(self, hsearch_params: list = []) -> None:
        """Run sequential hyperparameter search."""

        hypersearch_subroutine(hsearch_params, self.model, self.X_train, self.y_train)

    def feature_reduction(self, sl_params: dict = {}) -> None:
        """Reduce trained model by feature importances of xgb model."""

        feature_reduction_subroutine(
            sl_params, self.model, self.X_train, self.y_train, self.X_test, self.y_test
        )

    def save_model(self, sl_params: dict = {}) -> None:
        """Save model."""

        save_model_subroutine(sl_params, self.model)


class lr_model(object):
    """Linear regression model."""

    def __init__(self, train_test_data: train_test_data_type) -> None:
        """Initialize class attributes for regression model."""

        # model
        self.model: Optional[sklearn.linear_model.ridge.Ridge] = None

        # train, test subsets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_data

    def train(self, lr_train_params: dict = {}, sl_params: dict = {}) -> None:
        """Train regression model."""

        # replace params
        lr_train_params = (
            ReplaceParams(self.model, replace_params=lr_train_params)
            if self.model
            else lr_train_params
        )

        # initiate model
        self.model = Ridge(**lr_train_params)

        # train
        train_subroutine(sl_params, self.model, self.X_train, self.y_train)

    def predict(self, sl_params: dict = {}) -> None:
        """Predict with trained regression model."""

        predict_subroutine(sl_params, self.model, self.X_test, self.y_test)

    def hypersearch(self, hsearch_params: list = []) -> None:
        """Run sequential hyperparameter search."""

        hypersearch_subroutine(hsearch_params, self.model, self.X_train, self.y_train)

    def save_model(self, sl_params: dict = {}) -> None:
        """Save model."""

        save_model_subroutine(sl_params, self.model)
