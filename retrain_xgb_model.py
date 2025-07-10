import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# 1. Load your data
# Make sure the Excel file is in the same directory as this script
# and has the columns: Age, Gender, Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB, Churn

df = pd.read_excel("customer_churn_large_dataset.xlsx")

# 2. Encode Gender
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)

# 3. One-hot encode Location (for Houston, Los Angeles, Miami, New York)
for loc in ["Houston", "Los Angeles", "Miami", "New York"]:
    df[f"Location_{loc}"] = (df['Location'] == loc).astype(int)

# 4. Scale features as in your app
df['Age_scaled'] = (df['Age'] - 18) / (70 - 18)
df['Subscription_Length_scaled'] = (df['Subscription_Length_Months'] - 1) / (24 - 1)
df['Monthly_Bill_scaled'] = (df['Monthly_Bill'] - 30) / (100 - 30)
df['Total_Usage_scaled'] = (df['Total_Usage_GB'] - 50) / (500 - 50)

# 5. Prepare feature matrix X and target y
feature_cols = [
    'Age_scaled', 'Subscription_Length_scaled', 'Monthly_Bill_scaled', 'Total_Usage_scaled',
    'Gender_Male', 'Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York'
]
X = df[feature_cols]
y = df['Churn']

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the XGBoost model
model = XGBClassifier(
    use_label_encoder=False,  # Suppress warning in new XGBoost
    eval_metric='logloss',    # Required for classification
    random_state=42
)
model.fit(X_train, y_train)

# 8. Save the model
joblib.dump(model, "customer_churn_classifier.pkl")
print("Model retrained and saved as customer_churn_classifier.pkl") 