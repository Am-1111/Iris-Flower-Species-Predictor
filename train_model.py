# train_model.py
# This script trains a simple machine learning model and saves it.
# Run this script once before you run the Streamlit app.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib # Used for saving and loading the model

def train_and_save_model():
    """
    Trains a Logistic Regression model on the Iris dataset and saves it.
    """
    # 1. Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    
    # For clarity, let's map target integers to species names
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    y = y.map(species_map)

    print("Training data loaded successfully.")
    print("Features (X):")
    print(X.head())
    print("\nTarget (y):")
    print(y.head())

    # 2. Initialize and train the model
    # We use a simple Logistic Regression model for this example.
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    
    print("\nModel training complete.")

    # 3. Save the trained model to a file
    # joblib is efficient for saving scikit-learn models.
    model_filename = 'iris_model.joblib'
    joblib.dump(model, model_filename)
    
    print(f"Model saved successfully as '{model_filename}'.")

if __name__ == '__main__':
    train_and_save_model()

