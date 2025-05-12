# tuning.py
"""
This module provides utility functions for model hyperparameter optimization using Optuna.
It supports flexible search spaces, multiple models, and customizable scoring metrics.
"""

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from optuna import Trial
from typing import Callable, Dict, Any, Tuple
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_model_with_optuna(
    model_class: Callable[..., Any],
    param_space: Callable[[Trial], Dict[str, Any]],
    X,
    y,
    scoring: str = "f1",
    n_trials: int = 50,
    timeout: int = None,
    direction: str = "maximize",
    cv: int = 5,
    random_state: int = 42,
    **model_kwargs
) -> Tuple[optuna.Study, Dict[str, Any], Any]:
    """
    Run Optuna optimization for a given model class and parameter space.

    Parameters:
        model_class: A scikit-learn-compatible model class (e.g. XGBClassifier).
        param_space: A function that defines the parameter search space for Optuna.
        X: Training features.
        y: Training targets.
        scoring: Scikit-learn scoring metric.
        n_trials: Number of Optuna trials.
        timeout: Optional timeout in seconds.
        direction: "maximize" or "minimize" the score.
        cv: Number of cross-validation folds.
        random_state: Seed for reproducibility.
        model_kwargs: Additional fixed arguments for the model.

    Returns:
        study: Optuna study object.
        best_params: Dictionary of the best parameters found.
        best_model: Trained model with the best parameters.
    """

    def objective(trial: Trial) -> float:
        params = param_space(trial)
        model = model_class(**params, **model_kwargs)
        score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
        return score

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))

    print(f"🔍 Starting hyperparameter optimization for {model_class.__name__} ({scoring})")
    start = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    elapsed = time.time() - start
    print(f"✅ Optimization finished in {elapsed:.1f} sec")
    print(f"Best params: {study.best_params}")

    best_model = model_class(**study.best_params, **model_kwargs)
    best_model.fit(X, y)

    return study, study.best_params, best_model

def create_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """
    Create parameter grid for different model types.
    
    Args:
        model_type: Type of model ('xgb', 'rf', etc.)
    
    Returns:
        Dictionary of parameter grids
    """
    if model_type == 'xgb':
        return {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    elif model_type == 'rf':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def grid_search_optimize(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform grid search optimization.
    
    Args:
        model: Model instance to optimize
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid for optimization
        cv: Number of cross-validation folds
        scoring: Scoring metric
    
    Returns:
        Tuple of (best model, best parameters)
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def random_search_optimize(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: Dict[str, List[Any]],
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform random search optimization.
    
    Args:
        model: Model instance to optimize
        X_train: Training features
        y_train: Training target
        param_distributions: Parameter distributions for optimization
        n_iter: Number of iterations
        cv: Number of cross-validation folds
        scoring: Scoring metric
    
    Returns:
        Tuple of (best model, best parameters)
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_

def optuna_optimize(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform optimization using Optuna.
    
    Args:
        model_type: Type of model ('xgb', 'rf', etc.)
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best model, best parameters)
    """
    def objective(trial):
        if model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'gamma': trial.suggest_float('gamma', 0, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params)
        elif model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        scores = []
        for i in range(cv):
            # Implement your cross-validation logic here
            # This is a simplified version
            train_idx = np.random.choice(len(X_train), size=int(0.8*len(X_train)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(X_train)), train_idx)
            
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred)
            scores.append(score)
        
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best score: {study.best_value:.4f}")
    
    # Create and return the best model
    if model_type == 'xgb':
        best_model = xgb.XGBClassifier(**study.best_params)
    elif model_type == 'rf':
        best_model = RandomForestClassifier(**study.best_params)
    
    best_model.fit(X_train, y_train)
    return best_model, study.best_params

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    logger.info("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")
    
    return metrics
