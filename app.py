from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_PATHS = [Path("model.joblib"), Path("model.pkl"), Path("models/churn_pipeline.joblib")]


@st.cache_resource
def load_saved_model(model_path: str):
    return joblib.load(model_path)


def resolve_default_model_path() -> Path | None:
    for model_path in DEFAULT_MODEL_PATHS:
        if model_path.exists():
            return model_path
    return None


def prepare_input_for_model(input_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Prepare UI input for models trained in notebook style (encoded numerics)
    or pipeline style (raw categoricals).
    """
    if isinstance(model, Pipeline):
        # Pipeline models handle raw categorical values internally.
        return input_df

    # Notebook-style encoding: Gender -> binary, Geography -> numeric codes.
    geography_map = {"France": 0, "Germany": 1, "Spain": 2}
    prepared_df = input_df.copy()
    prepared_df["Geography"] = prepared_df["Geography"].map(geography_map).fillna(0).astype(int)
    prepared_df["Gender"] = prepared_df["Gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)

    # Align feature order with trained estimator when available.
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        prepared_df = prepared_df[expected_cols]

    return prepared_df


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
    st.title("Customer Churn Prediction Interface")
    st.write("Use your saved model file to predict churn probability.")

    default_model = resolve_default_model_path()
    selected_path = st.text_input(
        "Model file path (.joblib or .pkl)",
        value=str(default_model) if default_model else "",
        help="Use a trained pipeline/classifier file saved from your notebook.",
    )

    if not selected_path:
        st.warning("Please enter your model file path to continue.")
        st.info(
            "Example notebook export:\n"
            "import joblib\n"
            "joblib.dump(model, 'model.joblib')"
        )
        st.stop()

    model_file = Path(selected_path)
    if not model_file.exists():
        st.error(f"Model file not found: {model_file}")
        st.stop()

    try:
        model = load_saved_model(str(model_file))
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    if not isinstance(model, Pipeline) and not hasattr(model, "predict_proba"):
        st.error("Loaded object is not a valid sklearn model/pipeline with predict_proba().")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=100.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda v: "Yes" if v == 1 else "No")
        is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda v: "Yes" if v == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=100.0)

    if st.button("Predict Churn"):
        input_df = pd.DataFrame(
            [
                {
                    "CreditScore": credit_score,
                    "Geography": geography,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": num_products,
                    "HasCrCard": has_cr_card,
                    "IsActiveMember": is_active_member,
                    "EstimatedSalary": estimated_salary,
                }
            ]
        )

        model_input = prepare_input_for_model(input_df, model)
        churn_probability = model.predict_proba(model_input)[0][1]
        churn_prediction = int(churn_probability >= 0.5)

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")

        if churn_prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is likely to stay.")


if __name__ == "__main__":
    main()
