import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from fpdf import FPDF
import base64
import os
import gdown

# --------------------------------
# 🔁 Load model from Google Drive
# --------------------------------
model_id = "1Uh_kgkaIVeRlZsvp6oHMW2_cKZZlosbu"
features_id = "12XJQ4vl1zf68Ou2EQg5far77M5NAzkx1"

if not os.path.exists("credit_model.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={model_id}", "credit_model.pkl", quiet=False)

if not os.path.exists("feature_names.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={features_id}", "feature_names.pkl", quiet=False)

# Load model and features
model = joblib.load("credit_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Predict", "🧠 Model Explanation", "📘 Disclaimer"])

# -----------------------------
# 🏠 Home Page
# -----------------------------
if page == "🏠 Home":
    st.title("Credit Default Prediction App")
    st.markdown("""
    Welcome to the Credit Default Risk Assessment tool.  
    This app predicts the probability that a borrower will default on credit.

    ### What’s inside?
    - A fairness-aware Random Forest model
    - Trained on 10,000 real + synthetic borrower profiles
    - No use of sensitive features like gender or marital status

    ⚠️ Use responsibly. This is not a financial advice tool.
    """)

# -----------------------------
# 🔮 Prediction Page
# -----------------------------
elif page == "🔮 Predict":
    st.title("Predict Credit Default Risk")

    with st.form("prediction_form"):
        st.markdown("### 📥 Enter Borrower Information")

        INCOME = st.number_input("INCOME", min_value=0)
        SAVINGS = st.number_input("SAVINGS", min_value=0)
        DEBT = st.number_input("DEBT", min_value=0)

        education = st.selectbox("Education Level", [
            "No formal education", "Primary", "Secondary", "High School", "Diploma",
            "Bachelor's", "Master's", "PhD"
        ])
        occupation = st.selectbox("Occupation", [
            "Unemployed", "Student", "Agriculture", "Manual labor", "Sales", "Clerical",
            "Skilled Trade", "Health Care", "Education", "Engineering/Tech", "Managerial",
            "Professional Services", "Self-employed", "Retired", "Other"
        ])
        relationship = st.selectbox("Household Role", [
            "Single", "Married", "Divorced", "Widowed", "Supporting dependents", "Living with family"
        ])

        threshold = st.slider("Set Risk Threshold", 0.0, 1.0, 0.4, 0.01)

        submitted = st.form_submit_button("🔮 Predict Default Risk")

    if submitted:
        if INCOME == 0 or SAVINGS == 0 or DEBT == 0:
            st.warning("⚠️ Some inputs are set to 0. This may lead to unrealistic predictions.")
        if INCOME == 0 and SAVINGS == 0 and DEBT == 0:
            st.error("❌ Please enter valid borrower financial details before predicting.")
        else:
            # Feature engineering
            R_DEBT_INCOME = DEBT / INCOME if INCOME > 0 else 0
            R_DEBT_SAVINGS = DEBT / SAVINGS if SAVINGS > 0 else 0
            CAT_DEBT = 1 if DEBT > 0 else 0
            CAT_SAVINGS_ACCOUNT = 1 if SAVINGS > 0 else 0

            input_dict = {
                'INCOME': INCOME,
                'SAVINGS': SAVINGS,
                'DEBT': DEBT,
                'R_DEBT_INCOME': R_DEBT_INCOME,
                'R_DEBT_SAVINGS': R_DEBT_SAVINGS,
                'CAT_DEBT': CAT_DEBT,
                'CAT_SAVINGS_ACCOUNT': CAT_SAVINGS_ACCOUNT,
                f'education_{education}': 1,
                f'occupation_{occupation}': 1,
                f'relationship_{relationship}': 1
            }

            full_input = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_names}])

            prob = model.predict_proba(full_input)[0][1]
            pred = 1 if prob >= threshold else 0

            # Risk Band logic
            if prob < 0.3:
                risk_band = "Low Risk"
            elif prob < 0.6:
                risk_band = "Medium Risk"
            else:
                risk_band = "High Risk"

            st.subheader("📊 Prediction Result")
            st.write(f"**Predicted Probability:** {prob:.2%}")
            st.write(f"**Classification:** {'High Risk ⚠️' if pred else 'Low Risk ✅'}")
            st.write(f"**Risk Level:** `{risk_band}`")

            if pred == 1:
                st.error("⚠️ High Risk: This borrower is likely to default.")
            else:
                st.success("✅ Low Risk: This borrower is unlikely to default.")

            # PDF Report
            def build_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Credit Default Risk Report", ln=True, align='C')
                pdf.ln(5)
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Borrower Information:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, f"""
INCOME: {INCOME}
SAVINGS: {SAVINGS}
DEBT: {DEBT}
Debt-to-Income Ratio: {R_DEBT_INCOME:.2f}
Debt-to-Savings Ratio: {R_DEBT_SAVINGS:.2f}
Education: {education}
Occupation: {occupation}
Relationship: {relationship}
""")

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Prediction Summary:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, f"""
Predicted Risk Probability: {prob:.2%}
Threshold Used: {threshold}
Final Classification: {'High Risk' if pred else 'Low Risk'}
Risk Level: {risk_band}
""")

                pdf.set_y(-30)
                pdf.set_font("Arial", size=8)
                pdf.multi_cell(0, 5, "Disclaimer: This prediction is based on statistical models and does not constitute financial advice or guarantee future performance.")
                return pdf

            pdf = build_pdf()
            pdf.output("borrower_report.pdf")

            with open("borrower_report.pdf", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="borrower_report.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# 🧠 Model Explanation Page
# -----------------------------
elif page == "🧠 Model Explanation":
    st.title("How the Model Works")
    st.markdown("""
This app uses a **Random Forest model** trained on 10,000 borrower records, including a mix of real and synthetic data.

We engineered important features like:
- Debt-to-Income Ratio
- Debt-to-Savings Ratio
- Binary indicators for having any debt or savings

### 📉 Fairness by Design
To reduce potential bias in credit scoring, we excluded sensitive variables such as:
- Gender
- Marital Status
- Relationship Type

### ✅ Validation
The model was rigorously evaluated using:
- Accuracy
- F1-Score
- Recall (important for identifying likely defaulters)
- AUC (Area Under the ROC Curve)

This ensures the model is not only accurate but generalizes well to unseen cases.

For responsible use, always combine model outputs with domain expertise.
""")

# -----------------------------
# 📘 Disclaimer Page
# -----------------------------
elif page == "📘 Disclaimer":
    st.title("Disclaimer & Contact")
    st.markdown("""
This tool is designed for **educational and exploratory** purposes only.

### ⚠️ Disclaimer:
- This app does **not replace** professional financial risk assessment.
- Model outputs are **probabilistic**, not deterministic.

### 📫 Contact:
For questions or feedback, reach out to: `regina.gathimba@strathmore.edu`
""")
