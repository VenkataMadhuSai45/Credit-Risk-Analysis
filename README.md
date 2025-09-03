# Credit Risk Modelling

This project provides a complete workflow for credit risk analysis using machine learning. It predicts loan approval likelihood based on customer data, with a focus on robust feature engineering and model selection.

## Features

- Data cleaning and preprocessing
- Feature selection using Chi-square, ANOVA, and VIF
- Handling categorical and numerical variables
- Preprocessing pipelines (imputation, scaling, encoding)
- Model training and evaluation with:
  - Logistic Regression
  - Decision Tree
  - SVM
  - Naive Bayes
  - Random Forest
  - AdaBoost
  - XGBoost
- Hyperparameter tuning with GridSearchCV
- Performance metrics and confusion matrix visualization

## Workflow

1. **Data Loading:** Reads and merges two Excel datasets on `PROSPECTID`.
2. **Preprocessing:** Removes columns/rows with excessive missing values, encodes categorical features, and scales numerical features.
3. **Feature Selection:** Uses statistical tests to select relevant features.
4. **Model Training:** Trains several classifiers and selects the best based on accuracy.
5. **Hyperparameter Tuning:** Optimizes the best model (XGBoost) using grid search.
6. **Evaluation:** Prints accuracy, classification report, and displays a confusion matrix.

## Usage

Open `Credit Risk Modelling.ipynb` and run the cells sequentially. Ensure the required datasets are available and update file paths if needed.

## Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, statsmodels, scipy

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels scipy
```

## Project Structure

- `Credit Risk Modelling.ipynb`: Main notebook with all code and analysis


## Results

- Multiple models are compared; the best is selected based on accuracy.
- XGBoost is further tuned for optimal performance.
- Evaluation includes accuracy, classification report, and confusion matrix visualization.



