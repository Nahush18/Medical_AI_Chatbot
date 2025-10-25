import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import os

# Cache the data to improve performance
@st.cache_data()
def load_data():
    """
    Load the diabetes dataset from a local file or a remote GitHub URL.
    Returns:
        df (DataFrame): Complete dataset
        X (DataFrame): Features for training
        y (Series): Target variable
    """
    # Try to load the dataset locally first
    local_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        # Fallback to GitHub raw link
        url = "https://raw.githubusercontent.com/Nahush18/Medical_AI_Chatbot/main/Diabetes-Healthcare-Programme-main/diabetes.csv"
        try:
            df = pd.read_csv(url)
        except Exception as e:
            st.error("❌ Failed to load the dataset from both local and remote sources.")
            st.stop()

    # Define features (X) and label (y)
    X = df[['HbA1c_level', 'Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df['Outcome']

    return df, X, y


# Cache model training to avoid retraining on every run
@st.cache_data()
def train_model(X, y):
    """
    Train a Decision Tree model on the given dataset.
    Returns:
        model: Trained DecisionTreeClassifier
        score: Training accuracy
    """
    model = DecisionTreeClassifier(
        ccp_alpha=0.0,               # Controls pruning (reduce overfitting)
        class_weight=None,           # Class weights (None = uniform)
        criterion='entropy',         # Measure of split quality
        max_depth=4,                 # Controls tree depth
        max_features=None,           # All features used for split
        max_leaf_nodes=None,         # No limit on leaf nodes
        min_impurity_decrease=0.0,   # Minimum impurity decrease for split
        min_samples_leaf=1,          # Minimum samples per leaf
        min_samples_split=2,         # Minimum samples to split
        min_weight_fraction_leaf=0.0,
        random_state=42,             # Reproducible results
        splitter='best'              # Best split strategy
    )

    model.fit(X, y)
    score = model.score(X, y)

    return model, score


def predict(X, y, features):
    """
    Predict the outcome using the trained model.
    Args:
        X: Training features
        y: Target variable
        features: Input features for prediction (list or array)
    Returns:
        prediction: Predicted class (0 or 1)
        score: Model training accuracy
    """
    model, score = train_model(X, y)

    # Ensure input features are in correct shape
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    return prediction, score
