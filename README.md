# Customer Churn Prediction

This project predicts whether a customer is likely to leave a bank service (`Exited = 1`) using machine learning.

The implementation is notebook-based and covers an end-to-end workflow:
- data loading and cleaning
- exploratory data analysis (EDA)
- feature preparation and encoding
- model training and tuning
- performance evaluation

## Project Files

```text
Project---Customer-Churn-Prediction/
├── Churn_Modelling.csv
├── Customer_Churn_Prediction.ipynb
├── app.py
└── README.md
```

## Dataset

Source file: `Churn_Modelling.csv`

Important columns:
- Customer/account features: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- Target variable: `Exited` (0 = stayed, 1 = churned)

## Notebook Workflow

The notebook `Customer_Churn_Prediction.ipynb` includes:
1. Data understanding (`head`, `shape`, `info`, null checks)
2. EDA and churn distribution analysis
3. Feature selection and preprocessing
4. Train-test split
5. Model training (including boosting models)
6. Hyperparameter tuning (GridSearchCV)
7. Evaluation using confusion matrix, precision, recall, F1-score, and accuracy

## How to Run

1. Open `Customer_Churn_Prediction.ipynb` in Jupyter Notebook or VS Code.
2. Install required libraries if not already installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

3. Run notebook cells from top to bottom.

## User Interface (Streamlit)

You can run a simple web interface for real-time prediction using your saved model.

First, save your trained model from notebook:

```python
import joblib
joblib.dump(model, "model.joblib")
```

Then start the app:

```bash
pip install streamlit
streamlit run app.py
```

What it does:
- loads your saved model file (`model.joblib` / `model.pkl`)
- lets users enter customer data using form inputs
- returns churn probability and final prediction

## Objective

The goal is to identify customers at risk of churn so businesses can take proactive retention actions and reduce revenue loss.

## Future Improvements

- Build a reusable training/inference pipeline outside the notebook
- Add model artifact saving and versioning
- Optimize decision threshold for better churn recall
- Add reproducible environment files (`requirements.txt` or `environment.yml`)
