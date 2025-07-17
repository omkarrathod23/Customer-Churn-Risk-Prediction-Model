import pandas as pd
from xgboost import XGBClassifier
import joblib

# 1. Load your data
# Replace this with your actual data loading code
# Example:
# df = pd.read_csv('your_data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

# For demonstration, here's a dummy dataset:
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, (iris.target == 0).astype(int)  # Binary target for example

# 2. Train the model (do NOT use use_label_encoder)
model = XGBClassifier()
model.fit(X, y)

# 3. Save the model
joblib.dump(model, "customer_churn_classifier.pkl")
print("Model retrained and saved as customer_churn_classifier.pkl") 