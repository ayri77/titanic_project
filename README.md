# Titanic Survival Prediction Project

## Overview
This project analyzes the Titanic dataset to predict passenger survival using machine learning techniques. The project follows a structured approach from data preparation to model optimization.

## Project Structure
```
titanic_project/
├── data/                  # Data directory
├── models/               # Saved model files
├── utils/                # Utility functions
│   ├── feature_engineering.py
│   ├── metrics.py
│   └── visualisation.py
├── notebooks/            # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_pipeline_and_submission.ipynb
│   └── 05_model_optimization.ipynb
├── config.py            # Configuration settings
└── requirements.txt     # Project dependencies
```

## Setup and Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow
1. **Data Preparation** (`01_data_preparation.ipynb`)
   - Data loading and initial cleaning
   - Feature engineering
   - Missing value handling

2. **Data Analysis** (`02_data_analysis.ipynb`)
   - Exploratory data analysis
   - Feature importance analysis
   - Correlation studies

3. **Model Training** (`03_model_training.ipynb`)
   - Model selection and training
   - Cross-validation
   - Performance evaluation

4. **Pipeline and Submission** (`04_pipeline_and_submission.ipynb`)
   - Final model pipeline
   - Prediction generation
   - Submission preparation

5. **Model Optimization** (`05_model_optimization.ipynb`)
   - Hyperparameter tuning
   - Model ensemble techniques
   - Performance optimization

## Key Features
- Comprehensive data preprocessing pipeline
- Advanced feature engineering
- Multiple model evaluation metrics
- Interactive visualizations
- Model interpretation using SHAP values

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- xgboost

## Contributing
Feel free to submit issues and enhancement requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
