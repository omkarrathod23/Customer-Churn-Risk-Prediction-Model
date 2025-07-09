import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
import base64
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# --- UI CONFIG ---
st.set_page_config(
    page_title="Customer Churn Risk Prediction Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOGO ---
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <img src='https://img.icons8.com/fluency/96/000000/customer-insight.png' width='60' style='margin-right: 20px;'>
        <h1 style='display: inline;'>Customer Churn Risk Prediction Model</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- TABS ---
tabs = st.tabs(["Single Prediction", "Batch Prediction", "EDA & Insights", "About"])

# --- ENCODING & SCALING HELPERS ---
# Update the list of all possible locations (from dataset unique values)
ALL_LOCATIONS = [
    "Houston", "Los Angeles", "Miami", "Chicago", "New York"
]

def preprocess_input(age, gender, location, subscription_length, monthly_bill, total_usage, model_type="xgb"):
    if model_type == "keras":
        # Only the 4 features, scaled, in the order used for Keras
        age_scaled = (age - 18) / (70 - 18)
        subscription_length_scaled = (subscription_length - 1) / (24 - 1)
        monthly_bill_scaled = (monthly_bill - 30) / (100 - 30)
        total_usage_scaled = (total_usage - 50) / (500 - 50)
        features = [monthly_bill_scaled, total_usage_scaled, age_scaled, subscription_length_scaled]
        return np.array(features).reshape(1, -1)
    else:
        # XGBoost: all 9 features (one-hot for locations, drop_first=True as in training)
        gender_male = 1 if gender == "Male" else 0
        # One-hot encode all except the first location (drop_first=True)
        locs = ["Houston", "Los Angeles", "Miami", "New York"]
        loc_dict = {f"Location_{l}": 0 for l in locs}
        if location in locs:
            loc_dict[f"Location_{location}"] = 1
        age_scaled = (age - 18) / (70 - 18)
        subscription_length_scaled = (subscription_length - 1) / (24 - 1)
        monthly_bill_scaled = (monthly_bill - 30) / (100 - 30)
        total_usage_scaled = (total_usage - 50) / (500 - 50)
        features = [
            age_scaled,
            subscription_length_scaled,
            monthly_bill_scaled,
            total_usage_scaled,
            gender_male,
            loc_dict["Location_Houston"],
            loc_dict["Location_Los Angeles"],
            loc_dict["Location_Miami"],
            loc_dict["Location_New York"]
        ]
        return np.array(features).reshape(1, -1)

@st.cache_resource(show_spinner=False)
def load_xgb_model():
    return joblib.load("customer_churn_classifier.pkl")

@st.cache_resource(show_spinner=False)
def load_keras_model():
    return load_model("ChurnClassifier.h5")

# --- PREDICTION FUNCTION ---
def predict_churn(features, model, model_type="xgb"):
    if model_type == "xgb":
        prob = model.predict_proba(features)[0, 1]
    else:
        prob = model.predict(features)[0][0]
    label = "Likely to Churn" if prob >= 0.5 else "Not Likely to Churn"
    return label, prob

def get_xgb_feature_importance(model):
    fmap = [
        "Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB",
        "Gender_Male", "Location_Houston", "Location_Los Angeles", "Location_Miami", "Location_New York"
    ]
    importances = model.feature_importances_
    return pd.DataFrame({"Feature": fmap, "Importance": importances})

# --- TEMPLATE FOR BATCH ---
def get_template():
    template = pd.DataFrame({
        "Age": [35],
        "Gender": ["Male"],
        "Location": ["Houston"],
        "Subscription_Length_Months": [12],
        "Monthly_Bill": [65.0],
        "Total_Usage_GB": [250.0]
    })
    return template

def get_table_download_link(df, filename="template.xlsx"):
    output = BytesIO()
    df.to_excel(output, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Excel Template</a>'

# --- TAB 1: SINGLE PREDICTION ---
with tabs[0]:
    st.subheader("Single Customer Prediction")
    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 70, 35, help="Customer's age (18-70)")
            subscription_length = st.slider("Subscription Length (Months)", 1, 24, 12, help="Number of months subscribed (1-24)")
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
            monthly_bill = st.number_input("Monthly Bill ($)", min_value=30.0, max_value=100.0, value=65.0, help="Monthly bill amount (30-100)")
        with col3:
            location = st.selectbox("Location", ALL_LOCATIONS, help="Customer's city")
            total_usage = st.number_input("Total Usage (GB)", min_value=50.0, max_value=500.0, value=250.0, help="Total data usage in GB (50-500)")
        model_choice = st.radio("Select Model", ("XGBoost (pkl)", "Neural Network (h5)"), help="Choose which model to use for prediction.")
        submitted = st.form_submit_button("Predict Churn")
    if submitted:
        if model_choice == "XGBoost (pkl)":
            features = preprocess_input(age, gender, location, subscription_length, monthly_bill, total_usage, model_type="xgb")
            model = load_xgb_model()
            label, prob = predict_churn(features, model, model_type="xgb")
        else:
            features = preprocess_input(age, gender, location, subscription_length, monthly_bill, total_usage, model_type="keras")
            model = load_keras_model()
            label, prob = predict_churn(features, model, model_type="keras")
        color = "red" if label == "Likely to Churn" else "green"
        st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)
        st.metric("Model Confidence", f"{prob:.2%}")
        if model_choice == "XGBoost (pkl)":
            st.write("#### Feature Importance (Global)")
            fi = get_xgb_feature_importance(model)
            st.bar_chart(fi.set_index("Feature"))
            # SHAP explainability
            st.write("#### Why this prediction? (SHAP Explanation)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            plt.figure(figsize=(8, 2))
            shap.summary_plot(shap_values, features, feature_names=fi["Feature"].tolist(), plot_type="bar", show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        st.info("Green = Not Likely to Churn, Red = Likely to Churn. Model confidence is the probability of churn.")

# --- TAB 2: BATCH PREDICTION ---
with tabs[1]:
    st.subheader("Batch Prediction (Upload .xlsx)")
    st.markdown(get_table_download_link(get_template(), "churn_template.xlsx"), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"], help="Upload a file with columns: Age, Gender, Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB")
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Validate columns
            required_cols = ["Age", "Gender", "Location", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB"]
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing columns. Required: {required_cols}")
            else:
                batch_features = []
                for _, row in df.iterrows():
                    if st.session_state.get("batch_model_choice") == "Neural Network (h5)":
                        batch_features.append(preprocess_input(
                            row["Age"],
                            row["Gender"],
                            row["Location"],
                            row["Subscription_Length_Months"],
                            row["Monthly_Bill"],
                            row["Total_Usage_GB"],
                            model_type="keras"
                        )[0])
                    else:
                        batch_features.append(preprocess_input(
                            row["Age"],
                            row["Gender"],
                            row["Location"],
                            row["Subscription_Length_Months"],
                            row["Monthly_Bill"],
                            row["Total_Usage_GB"],
                            model_type="xgb"
                        )[0])
                batch_features = np.array(batch_features)
                if st.session_state.get("batch_model_choice") == "Neural Network (h5)":
                    model = load_keras_model()
                    probs = model.predict(batch_features).flatten()
                else:
                    model = load_xgb_model()
                    probs = model.predict_proba(batch_features)[:, 1]
                labels = np.where(probs >= 0.5, "Likely to Churn", "Not Likely to Churn")
                df_result = df.copy()
                df_result["Prediction"] = labels
                df_result["Confidence"] = probs
                st.dataframe(df_result)
                # Summary
                churn_pct = (labels == "Likely to Churn").sum() / len(labels) * 100
                st.success(f"{churn_pct:.1f}% of uploaded customers are likely to churn.")
                # Download link
                csv = df_result.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                st.markdown(f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv">Download Results as CSV</a>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    # Model choice for batch
    st.radio("Select Model for Batch Prediction", ("XGBoost (pkl)", "Neural Network (h5)"), key="batch_model_choice")

# --- TAB 3: EDA & INSIGHTS ---
with tabs[2]:
    st.subheader("Exploratory Data Analysis & Insights")
    st.markdown("Upload a sample of your data to see churn rates by segment.")
    eda_file = st.file_uploader("Upload Excel File for EDA", type=["xlsx"], key="eda_file")
    if eda_file is not None:
        try:
            eda_df = pd.read_excel(eda_file)
            if "Churn" not in eda_df.columns:
                st.error("Your file must include a 'Churn' column (1=churned, 0=retained).")
            else:
                st.write("### Churn Rate by Gender")
                st.bar_chart(eda_df.groupby("Gender")["Churn"].mean())
                st.write("### Churn Rate by Location")
                st.bar_chart(eda_df.groupby("Location")["Churn"].mean())
                st.write("### Churn Rate by Subscription Length")
                st.line_chart(eda_df.groupby("Subscription_Length_Months")["Churn"].mean())
                st.write("### Churn Rate by Age Group")
                eda_df["AgeGroup"] = pd.cut(eda_df["Age"], bins=[18, 30, 40, 50, 60, 70], labels=["18-30", "31-40", "41-50", "51-60", "61-70"])
                st.bar_chart(eda_df.groupby("AgeGroup")["Churn"].mean())
        except Exception as e:
            st.error(f"Error processing EDA file: {e}")
    else:
        st.info("Upload a sample data file to see interactive churn insights by segment.")

# --- TAB 4: ABOUT ---
with tabs[3]:
    st.subheader("About This Project")
    st.markdown("""
    **Customer Churn Risk Prediction Model** is a professional dashboard to help businesses identify customers at risk of leaving, using advanced machine learning models. 
    
    **Features:**
    - Predict churn for individual customers or in batch
    - Model selection: XGBoost or Neural Network
    - SHAP explainability for transparency
    - Batch upload with downloadable results and summary
    - Interactive EDA for business insights
    - Clean, modern UI with tooltips and error handling
    
    **How it works:**
    1. Enter customer details or upload a file
    2. Select a model
    3. Get instant predictions and insights
    
    **Data Privacy:** Uploaded data is processed in-memory and not stored.
    
    **Contact:** For questions or feedback, connect on [GitHub](https://github.com/omkarrathod23/Customer-Churn-Risk-Prediction-Model.git) or [LinkedIn](https://www.linkedin.com/in/omkar-rathod-a93467251/).
    """)

# --- THEME CUSTOMIZATION ---
st.markdown("""
<style>
    .stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 8px;}
    .stMetric {font-size: 20px;}
    .stTabs [data-baseweb="tab"] {font-size: 18px;}
</style>
""", unsafe_allow_html=True)

# ------------------------- FOOTER -------------------------
st.markdown("""
---
Made with ❤️ by Omkar Rathod | [GitHub](https://github.com/omkarrathod23/Customer-Churn-Risk-Prediction-Model.git) | [LinkedIn](https://www.linkedin.com/in/omkar-rathod-a93467251/)
""")