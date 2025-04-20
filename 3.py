import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Train a Model
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 2. Save the Model
# Save the trained model to a file
model_filename = 'iris_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# 3. Reuse the Model
# Load the saved model
loaded_model = joblib.load(model_filename)

# Use the loaded model to make predictions
new_predictions = loaded_model.predict(X_test)
print(f"Predictions using the loaded model: {new_predictions}")
