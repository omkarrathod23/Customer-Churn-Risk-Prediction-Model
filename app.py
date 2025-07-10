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
import plotly.graph_objects as go

# --- UI CONFIG ---
st.set_page_config(
    page_title="Customer Churn Risk Prediction Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL STYLING ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: #b3c7f9;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        text-shadow: 0 2px 8px #222a3a, 0 1px 0 #fff2;
    }
    
    .header-subtitle {
        color: #b3c7f9;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
        text-shadow: 0 1px 4px #222a3a;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #232837;
        color: #b3c7f9;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea;
        color: #fff;
    }
    
    /* Form styling */
    .stForm {
        background: #232837;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.18);
        border: 1px solid #2d3347;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #dbeafe;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 32px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-shadow: 0 1px 4px #222a3a;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: #232837;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.18);
        border: 1px solid #2d3347;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: #b3c7f9;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-links {
        margin: 1rem 0;
    }
    
    .footer-links a {
        color: #b3c7f9;
        text-decoration: none;
        margin: 0 10px;
        transition: color 0.3s ease;
    }
    
    .footer-links a:hover {
        color: #fff;
    }
    
    /* Success/Error message styling */
    .success-message {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Chart styling */
    .chart-container {
        background: #232837;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.18);
        margin: 1rem 0;
    }
    body, .main, .block-container {
        color: #e0e6ed !important;
        background: #181c23 !important;
    }
    label, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #e0e6ed !important;
        font-weight: 500;
        font-size: 1rem;
        text-shadow: 0 1px 4px #222a3a;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background: #232837 !important;
        color: #f1f5f9 !important;
        border-radius: 6px;
        border: 1px solid #2d3347;
    }
    .stSlider > div {
        color: #e0e6ed !important;
    }
</style>
""", unsafe_allow_html=True)

# --- BEST-IN-CLASS HERO/HOME BOX ---
st.markdown("""
<style>
.hero-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    min-height: 300px;
    max-width: 1100px;
    margin: 1rem auto 2.5rem auto;
    padding: 2.5rem 2.5rem 2.5rem 2rem;
    gap: 2.5rem;
    background: rgba(120, 120, 255, 0.16);
    border-radius: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    backdrop-filter: blur(14px);
    border: 1.5px solid rgba(120,120,255,0.18);
    position: relative;
    overflow: hidden;
}
.hero-accent {
    position: absolute;
    left: 0; top: 0;
    width: 8px; height: 100%;
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    border-radius: 32px 0 0 32px;
    box-shadow: 0 0 24px #667eea44;
}
.hero-left {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    color: #fff;
    animation: fadeInLeft 1.1s cubic-bezier(.77,0,.18,1) 0.1s both;
    z-index: 2;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    margin-bottom: 1.1rem;
    text-shadow: 0 4px 24px #222a3a, 0 1px 0 #fff2;
    letter-spacing: 1px;
    line-height: 1.1;
}
.hero-title .highlight {
    color: #3fa7ff;
    background: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
}
.hero-desc {
    font-size: 1.13rem;
    font-weight: 400;
    color: #e0e6ed;
    margin-bottom: 1.5rem;
    max-width: 420px;
    line-height: 1.7;
}
.hero-image {
    flex: 1.1;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeInRight 1.1s cubic-bezier(.77,0,.18,1) 0.3s both;
    z-index: 2;
}
.hero-image img {
    width: 95%;
    max-width: 520px;
    min-width: 260px;
    height: auto;
    border-radius: 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    object-fit: cover;
    filter: drop-shadow(0 2px 16px #667eea33);
    transition: transform 0.3s cubic-bezier(.77,0,.18,1);
}
.hero-image img:hover {
    transform: scale(1.04) rotate(-2deg);
    box-shadow: 0 12px 40px #ffd16644;
}
@media (max-width: 900px) {
    .hero-container {
        flex-direction: column;
        padding: 1.2rem 0.5rem;
        min-height: unset;
        max-width: 98vw;
    }
    .hero-image img {
        width: 90vw;
        margin-top: 1.2rem;
        min-width: unset;
        max-width: 98vw;
    }
    .hero-left {
        align-items: center;
        text-align: center;
    }
    .hero-accent {
        display: none;
    }
}
@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-40px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeInRight {
    from { opacity: 0; transform: translateX(40px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
<script></script>
<div class="hero-container" id="home">
    <div class="hero-accent"></div>
    <div class="hero-left">
        <div class="hero-title">Customer <span class="highlight">Churn</span> Risk Prediction Model</div>
        <div class="hero-desc">
            Predict and understand customer churn with advanced machine learning.<br>
            Instantly identify at-risk customers, analyze churn drivers, and boost retention with actionable insights.<br><br>
            <b>Empower your business with data-driven decisions.</b>
        </div>
    </div>
    <div class="hero-image">
        <img src="https://cdn.smartkarrot.com/wp-content/uploads/2020/10/Customer-Attrition.png" alt="Customer Churn Illustration">
    </div>
</div>
""", unsafe_allow_html=True)

# --- ENCODING & SCALING HELPERS ---
ALL_LOCATIONS = [
    "Houston", "Los Angeles", "Miami", "Chicago", "New York"
]

def preprocess_input(age, gender, location, subscription_length, monthly_bill, total_usage, model_type="xgb"):
    if model_type == "keras":
        age_scaled = (age - 18) / (70 - 18)
        subscription_length_scaled = (subscription_length - 1) / (24 - 1)
        monthly_bill_scaled = (monthly_bill - 30) / (100 - 30)
        total_usage_scaled = (total_usage - 50) / (500 - 50)
        features = [monthly_bill_scaled, total_usage_scaled, age_scaled, subscription_length_scaled]
        return np.array(features).reshape(1, -1)
    else:
        gender_male = 1 if gender == "Male" else 0
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
    try:
        return joblib.load("customer_churn_classifier.pkl")
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_keras_model():
    try:
        return load_model("ChurnClassifier.h5")
    except Exception as e:
        st.error(f"Error loading Neural Network model: {e}")
        return None

# --- PREDICTION FUNCTION ---
def predict_churn(features, model, model_type="xgb"):
    try:
        if model_type == "xgb":
            prob = model.predict_proba(features)[0, 1]
        else:
            prob = model.predict(features)[0][0]
        label = "Likely to Churn" if prob >= 0.5 else "Not Likely to Churn"
        return label, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

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
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="color: #667eea; text-decoration: none; font-weight: 600;">📥 Download Excel Template</a>'

# --- TABS ---
tabs = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Analytics & Insights", "ℹ️ About"])

# --- TAB 1: SINGLE PREDICTION ---
with tabs[0]:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin: 0; color: #667eea;">Individual Customer Analysis</h3>
        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Enter customer details to predict churn risk and get detailed insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        model_choice = st.radio("Select Model", ("XGBoost (Recommended)", "Neural Network"), help="Choose which model to use for prediction.")
        submitted = st.form_submit_button("🚀 Predict Churn Risk")
    
    if submitted:
        with st.spinner("Analyzing customer data..."):
            if model_choice == "XGBoost (Recommended)":
                features = preprocess_input(age, gender, location, subscription_length, monthly_bill, total_usage, model_type="xgb")
                model = load_xgb_model()
                if model is not None:
                    label, prob = predict_churn(features, model, model_type="xgb")
            else:
                features = preprocess_input(age, gender, location, subscription_length, monthly_bill, total_usage, model_type="keras")
                model = load_keras_model()
                if model is not None:
                    label, prob = predict_churn(features, model, model_type="keras")
            
            if label and prob is not None:
                # Professional result display
                if label == "Likely to Churn":
                    st.markdown("""
                    <div class="error-message">
                        <h3 style="margin: 0;">⚠️ High Churn Risk Detected</h3>
                        <p style="margin: 0.5rem 0 0 0;">This customer shows signs of potential churn. Consider retention strategies.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-message">
                        <h3 style="margin: 0;">✅ Low Churn Risk</h3>
                        <p style="margin: 0.5rem 0 0 0;">This customer appears to be stable and satisfied.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Professional metric display
                st.markdown("""
                <div class="metric-container">
                    <h4 style="margin: 0; color: #667eea;">Model Confidence</h4>
                    <div class="metric-value">{:.1%}</div>
                </div>
                """.format(prob), unsafe_allow_html=True)
                
                if model_choice == "XGBoost (Recommended)" and model is not None:
                    # Feature importance with Plotly
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="margin: 0 0 1rem 0; color: #667eea;">Feature Importance Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    fi = get_xgb_feature_importance(model)
                    fig = go.Figure(go.Bar(
                        x=fi['Importance'],
                        y=fi['Feature'],
                        orientation='h',
                        marker=dict(color='#3fa7ff'),
                        text=[f"{v:.2f}" for v in fi['Importance']],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        height=320,
                        width=520,
                        margin=dict(l=80, r=20, t=40, b=40),
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        title=dict(text="Feature Importance", x=0.5, font=dict(size=18)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    fig.update_xaxes(showgrid=False, zeroline=False)
                    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed")
                    st.plotly_chart(fig, use_container_width=False)
                    
                    # SHAP explainability
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="margin: 0 0 1rem 0; color: #667eea;">Prediction Explanation (SHAP)</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(features)
                        plt.figure(figsize=(10, 4))
                        shap.summary_plot(shap_values, features, feature_names=fi["Feature"].tolist(), plot_type="bar", show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {e}")

# --- TAB 2: BATCH PREDICTION ---
with tabs[1]:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin: 0; color: #667eea;">Bulk Customer Analysis</h3>
        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Upload customer data in Excel format for batch churn prediction and analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(get_table_download_link(get_template(), "churn_template.xlsx"), unsafe_allow_html=True)
    with col2:
        batch_model_choice = st.radio("Model Selection", ("XGBoost", "Neural Network"), key="batch_model_choice")
    
    uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"], help="Upload a file with columns: Age, Gender, Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing customer data..."):
                df = pd.read_excel(uploaded_file)
                required_cols = ["Age", "Gender", "Location", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB"]
                
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ Missing required columns. Please ensure your file contains: {', '.join(required_cols)}")
                else:
                    # Data validation
                    invalid_data = []
                    for idx, row in df.iterrows():
                        if not (18 <= row["Age"] <= 70):
                            invalid_data.append(f"Row {idx+1}: Age must be between 18-70")
                        if row["Gender"] not in ["Male", "Female"]:
                            invalid_data.append(f"Row {idx+1}: Gender must be Male or Female")
                        if row["Location"] not in ALL_LOCATIONS:
                            invalid_data.append(f"Row {idx+1}: Invalid location")
                    
                    if invalid_data:
                        st.error("❌ Data validation errors:\n" + "\n".join(invalid_data[:5]))
                        if len(invalid_data) > 5:
                            st.error(f"... and {len(invalid_data)-5} more errors")
                    elif not invalid_data:
                        # Process predictions
                        batch_features = []
                        for _, row in df.iterrows():
                            if batch_model_choice == "Neural Network":
                                batch_features.append(preprocess_input(
                                    row["Age"], row["Gender"], row["Location"],
                                    row["Subscription_Length_Months"], row["Monthly_Bill"], row["Total_Usage_GB"],
                                    model_type="keras"
                                )[0])
                            else:
                                batch_features.append(preprocess_input(
                                    row["Age"], row["Gender"], row["Location"],
                                    row["Subscription_Length_Months"], row["Monthly_Bill"], row["Total_Usage_GB"],
                                    model_type="xgb"
                                )[0])
                        
                        batch_features = np.array(batch_features)
                        
                        if batch_model_choice == "Neural Network":
                            model = load_keras_model()
                            if model is not None:
                                probs = model.predict(batch_features).flatten()
                        else:
                            model = load_xgb_model()
                            if model is not None:
                                probs = model.predict_proba(batch_features)[:, 1]
                        
                        if model is not None:
                            labels = np.where(probs >= 0.5, "Likely to Churn", "Not Likely to Churn")
                            df_result = df.copy()
                            df_result["Prediction"] = labels
                            df_result["Confidence"] = [f"{p:.1%}" for p in probs]
                            
                            # Professional results display
                            st.markdown("""
                            <div class="metric-container">
                                <h4 style="margin: 0; color: #667eea;">Batch Analysis Results</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.dataframe(df_result, use_container_width=True)
                            
                            # Summary statistics
                            churn_count = (labels == "Likely to Churn").sum()
                            total_count = len(labels)
                            churn_pct = churn_count / total_count * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Customers", total_count)
                            with col2:
                                st.metric("At Risk", churn_count)
                            with col3:
                                st.metric("Risk Rate", f"{churn_pct:.1f}%")
                            
                            # Download results
                            csv = df_result.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv" style="color: #667eea; text-decoration: none; font-weight: 600;">📥 Download Results as CSV</a>', unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# --- TAB 3: EDA & INSIGHTS ---
with tabs[2]:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin: 0; color: #667eea;">Customer Analytics Dashboard</h3>
        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Upload historical data to analyze churn patterns and customer segments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    eda_file = st.file_uploader("📊 Upload Historical Data", type=["xlsx"], key="eda_file")
    
    if eda_file is not None:
        try:
            with st.spinner("Analyzing customer data..."):
                eda_df = pd.read_excel(eda_file)
                
                if "Churn" not in eda_df.columns:
                    st.error("❌ Your file must include a 'Churn' column (1=churned, 0=retained).")
                else:
                    st.markdown("""
                    <div class="chart-container">
                        <h4 style="margin: 0; color: #667eea;">Churn Analysis by Demographics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Churn Rate by Gender**")
                        gender_churn = eda_df.groupby("Gender")["Churn"].mean()
                        st.bar_chart(gender_churn)
                    
                    with col2:
                        st.markdown("**Churn Rate by Location**")
                        location_churn = eda_df.groupby("Location")["Churn"].mean()
                        st.bar_chart(location_churn)
                    
                    st.markdown("**Churn Rate by Subscription Length**")
                    subscription_churn = eda_df.groupby("Subscription_Length_Months")["Churn"].mean()
                    st.line_chart(subscription_churn)
                    
                    st.markdown("**Churn Rate by Age Group**")
                    eda_df["AgeGroup"] = pd.cut(eda_df["Age"], bins=[18, 30, 40, 50, 60, 70], labels=["18-30", "31-40", "41-50", "51-60", "61-70"])
                    age_churn = eda_df.groupby("AgeGroup")["Churn"].mean()
                    st.bar_chart(age_churn)
                    
                    # Summary statistics
                    total_customers = len(eda_df)
                    churned_customers = eda_df["Churn"].sum()
                    overall_churn_rate = churned_customers / total_customers * 100
                    
                    st.markdown("""
                    <div class="metric-container">
                        <h4 style="margin: 0; color: #667eea;">Overall Statistics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", total_customers)
                    with col2:
                        st.metric("Churned Customers", churned_customers)
                    with col3:
                        st.metric("Overall Churn Rate", f"{overall_churn_rate:.1f}%")
                        
        except Exception as e:
            st.error(f"❌ Error processing EDA file: {str(e)}")
    else:
        st.info("📊 Upload a historical data file to see interactive churn insights by segment.")

# --- TAB 4: ABOUT ---
with tabs[3]:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin: 0; color: #667eea;">About This Platform</h3>
        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Professional customer churn prediction and analytics solution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 **Platform Overview**
        
        **Customer Churn Risk Prediction Model** is an enterprise-grade analytics platform designed to help businesses identify and retain customers at risk of churning. Our advanced machine learning models provide accurate predictions and actionable insights.
        
        ### 🚀 **Key Features**
        
        - **🔍 Individual Analysis**: Predict churn risk for single customers with detailed explanations
        - **📊 Batch Processing**: Analyze large customer datasets efficiently
        - **📈 Advanced Analytics**: Interactive dashboards for customer segmentation
        - **🤖 Dual Models**: Choose between XGBoost and Neural Network algorithms
        - **🔍 Explainable AI**: SHAP explanations for transparent predictions
        - **📱 Professional UI**: Modern, responsive interface with intuitive design
        
        ### 🛡️ **Data Security**
        
        - All data processing occurs in-memory
        - No customer data is stored or transmitted
        - Secure, enterprise-grade infrastructure
        - GDPR compliant data handling
        
        ### 📊 **Model Performance**
        
        - **XGBoost**: High accuracy with interpretable feature importance
        - **Neural Network**: Deep learning approach for complex patterns
        - **Real-time Predictions**: Instant results with confidence scores
        - **Batch Processing**: Handle thousands of customers efficiently
        """)
    
    with col2:
        st.markdown("""
        ### 📋 **How to Use**
        
        1. **Single Prediction**: Enter customer details and get instant risk assessment
        2. **Batch Analysis**: Upload Excel files for bulk processing
        3. **Analytics**: Upload historical data for segment analysis
        4. **Download Results**: Export predictions and insights
        
        ### 🎓 **Technical Stack**
        
        - **Frontend**: Streamlit
        - **ML Models**: XGBoost, TensorFlow/Keras
        - **Analytics**: SHAP, Pandas, NumPy
        - **Visualization**: Matplotlib, Plotly
        
        ### 📞 **Support**
        
        For technical support or feature requests, please contact our team.
    """)

# --- PROFESSIONAL FOOTER ---
st.markdown("""
<div class="footer">
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div style="font-size: 20px; color: white; margin-bottom: 10px;">
            Made with <span style="color: #e74c3c;">❤️</span> by <strong>Omkar Rathod</strong>
    </div>
        <div class="footer-links">
            <a href="https://github.com/omkarrathod23" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" width="28" style="vertical-align: middle;">
        </a>
            <a href="https://www.linkedin.com/in/omkar-rathod-a93467251/" target="_blank">
                <img src="https://img.icons8.com/ios-filled/30/ffffff/linkedin.png" width="28" style="vertical-align: middle;">
            </a>
        </div>
        <div style="font-size: 14px; color: #bdc3c7; margin-top: 10px;">
            Customer Churn Risk Prediction Model &copy; 2025 | Enterprise Analytics Platform
        </div>
    </div>
</div>
""", unsafe_allow_html=True)