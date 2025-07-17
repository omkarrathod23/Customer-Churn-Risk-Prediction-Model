import pandas as pd
from xgboost import XGBClassifier
import joblib

# 1. Load your data
# Make sure the Excel file is in the same directory as this script
# and has the columns: Age, Gender, Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB, Churn
df = pd.read_excel("customer_churn_large_dataset.xlsx")

# 2. Preprocess (same as in your app)
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
locs = ["Houston", "Los Angeles", "Miami", "New York"]
for loc in locs:
    df[f"Location_{loc}"] = (df['Location'] == loc).astype(int)
df['Age_scaled'] = (df['Age'] - 18) / (70 - 18)
df['Subscription_Length_scaled'] = (df['Subscription_Length_Months'] - 1) / (24 - 1)
df['Monthly_Bill_scaled'] = (df['Monthly_Bill'] - 30) / (100 - 30)
df['Total_Usage_scaled'] = (df['Total_Usage_GB'] - 50) / (500 - 50)

# 3. Prepare features for XGBoost model (as in your app.py preprocess_input for xgb)
feature_cols = [
    'Age_scaled', 'Subscription_Length_scaled', 'Monthly_Bill_scaled', 'Total_Usage_scaled',
    'Gender_Male', 'Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York'
]
X = df[feature_cols]
y = df['Churn']

# 4. Train the XGBoost model (do NOT use use_label_encoder)
model = XGBClassifier()
model.fit(X, y)

# 5. Save the model
joblib.dump(model, "customer_churn_classifier.pkl")
print("XGBoost model retrained and saved as customer_churn_classifier.pkl") 