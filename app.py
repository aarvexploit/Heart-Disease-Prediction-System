import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load("models/heart_models.pkl")


artifacts = load_artifacts()
log_model = artifacts["log_model"]
tree_model = artifacts["tree_model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]

# Top feature importances from Decision Tree evaluation
top_importances = pd.Series(
    {
        "exang": 0.438,
        "cp_atypical angina": 0.195,
        "sex": 0.105,
        "age": 0.103,
        "thalch": 0.052,
        "oldpeak": 0.036,
    },
    name="Importance",
)

st.title("Heart Disease Prediction")
st.caption("Compare Logistic Regression and Decision Tree predictions on patient inputs")

# Sidebar with top feature importances
with st.sidebar:
    st.header("Top Decision Tree Features")
    st.bar_chart(top_importances)
    st.dataframe(top_importances.to_frame(), use_container_width=True)

st.subheader("Enter Patient Details")
col1, col2, col3 = st.columns(3)

with col1:
    exang = st.selectbox("Exercise-Induced Angina (exang)", ["No", "Yes"])
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        ["asymptomatic", "atypical angina", "non-anginal", "typical angina"],
    )
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=54)

with col2:
    thalch = st.number_input(
        "Max Heart Rate Achieved (thalch)", min_value=40.0, max_value=250.0, value=145.0
    )
    oldpeak = st.number_input(
        "ST Depression (oldpeak)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps)", min_value=50.0, max_value=260.0, value=130.0
    )
    chol = st.number_input(
        "Cholesterol (chol)", min_value=50.0, max_value=700.0, value=220.0
    )

with col3:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["False", "True"])
    restecg = st.selectbox(
        "Resting ECG (restecg)",
        ["lv hypertrophy", "normal", "st-t abnormality"],
    )
    slope = st.selectbox(
        "ST Slope (slope)",
        ["downsloping", "flat", "unknown", "upsloping"],
    )

if st.button("Run Prediction", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": 1 if sex == "Male" else 0,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": 1 if fbs == "True" else 0,
                "thalch": thalch,
                "exang": 1 if exang == "Yes" else 0,
                "oldpeak": oldpeak,
                "cp": cp,
                "restecg": restecg,
                "slope": slope,
            }
        ]
    )

    encoded_input = pd.get_dummies(
        input_df, columns=["cp", "restecg", "slope"], dtype=int
    )
    aligned_input = encoded_input.reindex(columns=feature_columns, fill_value=0)
    scaled_input = scaler.transform(aligned_input)

    # Logistic Regression uses scaled features; Decision Tree uses unscaled features
    log_prob = float(log_model.predict_proba(scaled_input)[0][1])
    log_pred = int(log_model.predict(scaled_input)[0])

    tree_prob = float(tree_model.predict_proba(aligned_input)[0][1])
    tree_pred = int(tree_model.predict(aligned_input)[0])

    st.divider()
    st.subheader("Model Predictions")

    res_left, res_right = st.columns(2)

    with res_left:
        st.markdown("### Logistic Regression")
        st.metric("Disease Probability", f"{log_prob:.1%}")
        if log_pred == 1:
            st.error("Predicted: Heart Disease Present")
        else:
            st.success("Predicted: No Heart Disease")

    with res_right:
        st.markdown("### Decision Tree")
        st.metric(
            "Disease Probability",
            f"{tree_prob:.1%}",
            delta=f"{(tree_prob - log_prob) * 100:+.1f}% vs Logistic Regression",
        )
        if tree_pred == 1:
            st.error("Predicted: Heart Disease Present")
        else:
            st.success("Predicted: No Heart Disease")

    with st.expander("Inspect Aligned Model Input Features"):
        st.dataframe(aligned_input, use_container_width=True, hide_index=True)