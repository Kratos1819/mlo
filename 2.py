import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Create output directory
os.makedirs('models', exist_ok=True)

# 1. Data Preparation
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

# 2. Model Training & 3. Hyperparameter Tuning
results = []
params = [0.01, 0.1, 1, 10, 100]  # Different values of C

for i, C_val in enumerate(params, start=1):
    model = LogisticRegression(C=C_val, max_iter=200)
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    model_filename = f"models/model_v{i}.pkl"
    joblib.dump(model, model_filename)

    # Store results
    results.append({
        'version': f"v{i}",
        'C': C_val,
        'accuracy': acc,
        'model_path': model_filename
    })

# 4. Record the Results
results_df = pd.DataFrame(results)
results_df.to_csv('models/model_results.csv', index=False)
print("Model training completed. Results:")
print(results_df)
