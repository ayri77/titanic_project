"""
Configuration settings for the Titanic Survival Prediction project.

This module contains all the constant values and configuration parameters
used throughout the project, including file paths, model parameters,
and other settings.
"""

# Data file paths
TRAIN_DATA_PATH = "./data/raw/train.csv"  # Path to the training dataset
TEST_DATA_PATH = "./data/raw/test.csv"    # Path to the test dataset
RESULTS_PATH = "./data/processed/"        # Directory for processed data and results

# Target variable
TARGET_NAME = "Survived"  # Name of the target variable in the dataset

# Model parameters
TEST_SIZE = 0.2          # Proportion of data to use for testing
RANDOM_STATE = 42        # Random seed for reproducibility

# Feature engineering parameters
AGE_BINS = [0, 3, 12, 18, 35, 60, 100]  # Age group boundaries
AGE_LABELS = ['baby', 'kids<3', 'kids<12', 'teenager', 'young', 'adult', 'senior']

# Model training parameters
CV_FOLDS = 5            # Number of cross-validation folds
EARLY_STOPPING_ROUNDS = 50  # Early stopping rounds for gradient boosting models