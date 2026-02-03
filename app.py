import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter

# Load model & metadata

with open("model.pkl", "rb") as f:
    cph = pickle.load(f)

with open("train_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Survival Analysis Dashboard", layout="wide")

st.title("🩺 Survival Prediction Dashboard (Cox Model)")

# Sidebar: Patient Input

st.sidebar.header("Patient Information")

input_data = {}

for col in feature_columns:
    input_data[col] = st.sidebar.number_input(
        col,
        value=0.0
    )

patient_df = pd.DataFrame([input_data])

# Risk Prediction

risk_score = cph.predict_partial_hazard(patient_df).iloc[0]
median_risk = np.median(
    cph.predict_partial_hazard(
        pd.DataFrame(np.zeros((100, len(feature_columns))), columns=feature_columns)
    )
)

risk_group = "High Risk" if risk_score >= median_risk else "Low Risk"

# Survival Function

surv_func = cph.predict_survival_function(patient_df)
median_survival_time = surv_func[surv_func.iloc[:, 0] <= 0.5].index.min()

# Layout

col1, col2 = st.columns(2)

# Left: Key Results
with col1:
    st.subheader("📊 Patient Risk Summary")

    st.metric("Risk Score", f"{risk_score:.3f}")
    st.metric("Risk Group", risk_group)
    st.metric(
        "Estimated Median Survival (Months)",
        f"{median_survival_time:.1f}" if pd.notnull(median_survival_time) else "Not reached"
    )

# Right: Survival Curve
with col2:
    st.subheader("📈 Predicted Survival Curve")

    fig, ax = plt.subplots()
    ax.step(
        surv_func.index,
        surv_func.iloc[:, 0],
        where="post"
    )
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Survival Probability")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

# Feature Contribution

st.subheader("🔍 Why this prediction? (Feature Contributions)")

coef = cph.params_
contributions = patient_df.iloc[0] * coef
contributions = contributions.sort_values(key=abs, ascending=False)

fig2, ax2 = plt.subplots(figsize=(8,4))
contributions.head(10).plot(kind="barh", ax=ax2)
ax2.set_title("Top Factors Influencing Risk")
ax2.set_xlabel("Contribution to Risk Score")
ax2.invert_yaxis()

st.pyplot(fig2)

# Explanation Text

st.markdown("### 🧠 Explanation")

top_features = contributions.head(5)

for feat, val in top_features.items():
    direction = "increases" if val > 0 else "reduces"
    st.write(
        f"- **{feat}** {direction} the risk due to its coefficient in the Cox model."
    )

st.info(
    "Risk is calculated using a penalized Cox proportional hazards model. "
    "Positive contributions increase hazard, negative contributions are protective."
)