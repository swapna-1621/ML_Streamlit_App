
# train.py
# Train a machine learning model to classify Iris flowers.

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the built-in Iris dataset
iris = load_iris()

# Separate flower measurements and their correct classes
X = iris.data
y = iris.target

# Create the machine learning model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Save the trained model into a file
joblib.dump(model, "iris_model.pkl")

print("Model trained and saved successfully!")
