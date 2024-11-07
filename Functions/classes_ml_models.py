import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from typing import Tuple, Dict, Any
from sklearn.cluster import KMeans


class BaseModel:
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame):
        """
        Base class for models

        Parameters:
        ----------
        data: pd.DataFrame
            The DataFrame containing the daily data
        signals: pd.DataFrame
            The DataFrame containing the daily signals
        """
        self.data_daily = data
        self.signals_daily = signals
        self.data_monthly = None
        self.signals_monthly = None
        self.scaler = None
        self.model = None

    def resample_monthly(self):
        """
        Resample daily data and signals to monthly frequency
        """
        # Resample data to get monthly closing prices
        self.data_monthly = self.data_daily.resample('M').last()

        # Resample signals to monthly timeframe
        self.signals_monthly = self.signals_daily.resample('M').last()

        # Ensure that indices align
        self.data_monthly = self.data_monthly.loc[self.signals_monthly.index]

    def prepare_features(self):
        """
        Prepare the feature matrix and target variable

        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _scale_data(self, X_train, X_test):
        """
        Scale the data using StandardScaler

        Parameters:
        ----------
        X_train: np.ndarray
            Training feature set
        X_test: np.ndarray
            Testing feature set

        Returns:
        -------
        X_train_scaled, X_test_scaled: Tuple[np.ndarray, np.ndarray]
            Scaled training and testing feature sets
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def _time_series_cv(self, n_splits: int):
        """
        Create a TimeSeriesSplit object

        Parameters:
        ----------
        n_splits: int
            Number of splits for cross-validation

        Returns:
        -------
        tscv: TimeSeriesSplit
            TimeSeriesSplit object
        """
        return TimeSeriesSplit(n_splits=n_splits)

    def fit(self):
        """
        Fit the model on the entire dataset

        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses should implement this method.")


class RidgeRegression(BaseModel):
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame):
        super().__init__(data, signals)
        self.X = None
        self.y = None

    def prepare_features(self):
        """
        Prepare the monthly features and target variable for regression
        """
        self.resample_monthly()

        # Calculate monthly returns as the target variable
        self.y = self.data_monthly.pct_change().shift(-1).dropna()

        # Align signals and target
        self.X = self.signals_monthly.shift(0).loc[self.y.index]

        # Drop any remaining NaN values
        self.X = self.X.dropna()
        self.y = self.y.loc[self.X.index]

    def fit(self, alpha: float):
        """
        Fit the Ridge Regression model on the entire dataset

        Parameters:
        ----------
        alpha: float
            Regularization strength
        """
        # Prepare features and target
        self.prepare_features()
        X_values = self.X.values
        y_values = self.y.values.flatten()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_values)

        # Initialize and fit the model
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y_values)

    def evaluate(self, n_splits: int = 5, alphas: np.ndarray = None) -> Tuple[float, float]:
        """
        Evaluate the Ridge Regression model using cross-validation and perform hyperparameter tuning

        Parameters:
        ----------
        n_splits: int
            Number of splits for cross-validation
        alphas: np.ndarray
            Array of alpha values to test

        Returns:
        -------
        best_alpha: float
            The alpha value that resulted in the highest R-squared score
        best_score: float
            The highest R-squared score achieved
        """
        if alphas is None:
            alphas = np.linspace(0, 3000, 100)

        self.prepare_features()
        X_values = self.X.values
        y_values = self.y.values.flatten()

        tscv = self._time_series_cv(n_splits)
        best_score = -np.inf  # Initialize to negative infinity for R-squared
        best_alpha = None

        for alpha in alphas:
            r2_scores = []
            for train_index, test_index in tscv.split(X_values):
                X_train, X_test = X_values[train_index], X_values[test_index]
                y_train, y_test = y_values[train_index], y_values[test_index]

                # Scale data within the fold
                X_train_scaled, X_test_scaled = self._scale_data(X_train, X_test)

                # Fit model
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)

            average_r2 = np.mean(r2_scores)
            if average_r2 > best_score:
                best_score = average_r2
                best_alpha = alpha

        # Fit the final model on the entire dataset using the best alpha
        self.fit(best_alpha)
        return best_alpha, best_score
    
    def get_feature_importances(self):
        """
        Get feature importances from the Ridge Regression model

        Returns:
        -------
        feature_importance_df: pd.DataFrame
            DataFrame containing feature names and their importance scores
        """
        # Ensure the model is trained
        if self.model is None:
            raise ValueError("Model has not been trained. Please call the 'fit' method before getting feature importances.")

        # Get coefficients
        coefficients = self.model.coef_

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': coefficients
        })

        return feature_importance_df
    

class MultipleLinearRegression(BaseModel):
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame):
        super().__init__(data, signals)
        self.X = None
        self.y = None

    def prepare_features(self):
        """
        Prepare the monthly features and target variable for regression
        """
        self.resample_monthly()

        # Calculate monthly returns as the target variable
        self.y = self.data_monthly.pct_change().shift(-1).dropna()

        # Align signals and target
        self.X = self.signals_monthly.shift(0).loc[self.y.index]

        # Drop any remaining NaN values
        self.X = self.X.dropna()
        self.y = self.y.loc[self.X.index]

    def fit(self):
        """
        Fit the Multiple Linear Regression model on the entire dataset
        """
        # Prepare features and target
        self.prepare_features()
        X_values = self.X.values
        y_values = self.y.values.flatten()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_values)

        # Initialize and fit the model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_values)

    def evaluate(self, n_splits: int = 5) -> float:
        """
        Evaluate the Multiple Linear Regression model using cross-validation

        Parameters:
        ----------
        n_splits: int
            Number of splits for cross-validation

        Returns:
        -------
        average_r2: float
            The average R-squared score over the cross-validation splits
        """
        self.prepare_features()
        X_values = self.X.values
        y_values = self.y.values.flatten()

        tscv = self._time_series_cv(n_splits)
        r2_scores = []

        for train_index, test_index in tscv.split(X_values):
            X_train, X_test = X_values[train_index], X_values[test_index]
            y_train, y_test = y_values[train_index], y_values[test_index]

            # Scale data within the fold
            X_train_scaled, X_test_scaled = self._scale_data(X_train, X_test)

            # Fit model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)

        average_r2 = np.mean(r2_scores)

        # Fit the final model on the entire dataset
        self.fit()

        return average_r2


class LassoRegression(BaseModel):
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame):
        super().__init__(data, signals)
        self.X = None
        self.y = None

    def prepare_features(self):
        """
        Prepare the monthly features and target variable for regression
        """
        self.resample_monthly()

        # Calculate monthly returns as the target variable
        self.y = self.data_monthly.pct_change().shift(-1).dropna()

        # Align signals and target
        self.X = self.signals_monthly.loc[self.y.index]

        # Drop any remaining NaN values
        self.X = self.X.dropna()
        self.y = self.y.loc[self.X.index]

    def fit(self, alpha: float):
        """
        Fit the Lasso Regression model on the entire dataset

        Parameters:
        ----------
        alpha: float
            Regularization strength
        """
        # Prepare features and target
        self.prepare_features()
        X_values = self.X.values
        y_values = self.y.values.flatten()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_values)

        # Initialize and fit the model
        self.model = Lasso(alpha=alpha, max_iter=10000)
        self.model.fit(X_scaled, y_values)

    def evaluate(self, n_splits: int = 5, alphas: np.ndarray = None) -> Tuple[float, float]:
        """
        Evaluate the Lasso Regression model using cross-validation and perform hyperparameter tuning

        Parameters:
        ----------
        n_splits: int
            Number of splits for cross-validation
        alphas: np.ndarray
            Array of alpha values to test

        Returns:
        -------
        best_alpha: float
            The alpha value that resulted in the highest R-squared score
        best_score: float
            The highest R-squared score achieved
        """
        if alphas is None:
            alphas = np.linspace(0.0, 3000, 100)

        self.prepare_features()
        X_values = self.X.values
        y_values = self.y.values.flatten()

        tscv = self._time_series_cv(n_splits)
        best_score = -np.inf  # Initialize to negative infinity for R-squared
        best_alpha = None

        for alpha in alphas:
            r2_scores = []
            for train_index, test_index in tscv.split(X_values):
                X_train, X_test = X_values[train_index], X_values[test_index]
                y_train, y_test = y_values[train_index], y_values[test_index]

                # Scale data within the fold
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Fit model
                model = Lasso(alpha=alpha, max_iter=10000)
                model.fit(X_train_scaled, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)

            average_r2 = np.mean(r2_scores)
            if average_r2 > best_score:
                best_score = average_r2
                best_alpha = alpha

        # Fit the final model on the entire dataset using the best alpha
        self.fit(best_alpha)
        return best_alpha, best_score


class RandomForestClassification(BaseModel):
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame):
        super().__init__(data, signals)
        self.X = None
        self.y = None

    def prepare_features(self):
        """
        Prepare the monthly features and target variable for classification.
        """
        self.resample_monthly()

        # Calculate monthly returns and shift it to align with current month signals
        returns = self.data_monthly.pct_change().shift(-1)

        # Create binary target variable: 1 if return > 0, else 0
        self.y = (returns > 0).astype(int).dropna()

        # Align signals and target
        self.X = self.signals_monthly.loc[self.y.index]

        # Drop any remaining NaN values
        self.X = self.X.dropna()
        self.y = self.y.loc[self.X.index].values.flatten()

    def fit(self, params: Dict[str, Any]):
        """
        Fit the Random Forest model.

        Parameters:
        ----------
        params: Dict[str, Any]
            Hyperparameters for the Random Forest classifier
        """
        # Prepare features and target
        self.prepare_features()
        X_values = self.X.values
        y_values = self.y

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_values)

        # Initialize the Random Forest model with given parameters
        self.model = RandomForestClassifier(**params, random_state=42)

        # Fit the model
        self.model.fit(X_scaled, y_values)

    def evaluate(self, n_splits: int = 5, param_grid: Dict[str, list] = None, scoring: str = 'neg_log_loss') -> Tuple[Dict[str, Any], float]:
        """
        Evaluate the Random Forest model using cross-validation and perform hyperparameter tuning

        Parameters:
        ----------
        n_splits: int
            Number of splits for cross-validation
        param_grid: Dict[str, list]
            Dictionary with parameters names as keys and lists of parameter settings to try as values
        scoring: str
            Scoring metric for evaluation, here we use negative log loss since it is the best (or one of the best) for probability output

        Returns:
        -------
        best_params: Dict[str, Any]
            The parameters that resulted in the best score
        best_score: float
            The best score achieved
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': [None, 'balanced']
            }

        self.prepare_features()
        X_values = self.X.values
        y_values = self.y

        tscv = TimeSeriesSplit(n_splits)
        rf_model = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1
        )

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_values)

        # Perform grid search
        grid_search.fit(X_scaled, y_values)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Fit the final model using the best parameters
        self.fit(best_params)

        return best_params, best_score

    def get_feature_importances(self):
        """
        Get feature importances from the Random Forest model

        Returns:
        -------
        feature_importance_df: pd.DataFrame
            DataFrame containing feature names and their importance scores
        """
        # Get feature importances
        importances = self.model.feature_importances_

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return feature_importance_df


class KMeansClustering():
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def k_means_clustering(self, k):
        """
        Perform K-means clustering using scikit-learn's KMeans on a pandas DataFrame

        Parameters:
            data: pandas DataFrame, shape=(n_samples, n_features)
            k: number of clusters

        Returns:
            kmeans: trained KMeans model
        """
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(self.data)
        return kmeans