import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

# 3. Prepare features for Keras model (as in your app.py preprocess_input for keras)
X = df[['Monthly_Bill_scaled', 'Total_Usage_scaled', 'Age_scaled', 'Subscription_Length_scaled']].values
y = df['Churn'].values

# 4. Build and train the model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(4,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32)

# 5. Save the model in HDF5 format
model.save("ChurnClassifier.h5")
print("Keras model retrained and saved as ChurnClassifier.h5") 