# Risk ML

Risk ML is a simple machine learning project focused on credit card fraud detection.  
The repository trains and compares multiple classification models using a highly imbalanced transaction dataset, applying SMOTE to improve class balance during training.

## Project overview

The main script:

- loads transaction data from `creditcard.csv`
- separates features from the `Class` target column
- splits the data into training and test sets
- scales the feature values with `StandardScaler`
- balances the training set with `SMOTE`
- trains and evaluates several machine learning models

## Models included

The current implementation compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbors

For each model, the script prints:

- training time
- classification report
- confusion matrix
- ROC AUC score

## Repository structure

- `fraud.py` — main training and evaluation script
- `README.md` — project documentation

## Requirements

Before running the project, make sure you have:

- Python 3 installed
- the dataset file `creditcard.csv` in the repository root

Install the required packages:

```bash
pip install pandas scikit-learn imbalanced-learn xgboost
```

## How to run

From the repository root, execute:

```bash
python fraud.py
```

## Expected input

The script expects a CSV file named `creditcard.csv` in the root of the project.  
It must contain:

- a `Class` column as the prediction target
- all other columns as numerical input features

## What this project is useful for

This project is a good starting point for:

- comparing baseline fraud detection models
- studying class imbalance handling with SMOTE
- evaluating model performance for binary classification problems
- experimenting with credit risk and anomaly detection workflows

## Current limitations

- the dataset path is hardcoded in `fraud.py`
- model hyperparameters use default values
- results are printed to the console only
- there is no persistence for trained models or metrics

## Next improvement ideas

- add a `requirements.txt` file
- make the dataset path configurable
- save metrics to a file
- add cross-validation and hyperparameter tuning
- include visualizations for model comparison
