import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import os

# -----------------------------
# Load dataset (robust for deployment)
# -----------------------------
@st.cache_data()
def load_data():
    """
    Load the diabetes dataset from local file or GitHub.
    Returns:
        df, X, y
    """
    import os
    import pandas as pd
    import streamlit as st

    # Use working directory instead of __file__
    local_path = os.path.join(os.getcwd(), "diabetes.csv")

    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            st.info("✅ Loaded dataset from local file.")
        else:
            # Fallback: GitHub raw link
            url = "https://raw.githubusercontent.com/Nahush18/Medical_AI_Chatbot/main/Diabetes-Healthcare-Programme-main/diabetes.csv"
            df = pd.read_csv(url)
            st.info("✅ Loaded dataset from GitHub URL.")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

    # Features and label
    X = df[['HbA1c_level', 'Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df['Outcome']

    return df, X, y



# -----------------------------
# Train Decision Tree Model
# -----------------------------
@st.cache_data()
def train_model(X, y):
    """
    Train a Decision Tree model on the given dataset.
    Returns:
        model: Trained DecisionTreeClassifier
        score: Training accuracy
    """
    model = DecisionTreeClassifier(
        ccp_alpha=0.0,                # Pruning control
        class_weight=None,            # Uniform weights
        criterion='entropy',          # Split quality
        max_depth=4,                  # Tree depth
        max_features=None,            # Use all features
        max_leaf_nodes=None,          # No limit on leaves
        min_impurity_decrease=0.0,    # Min impurity for split
        min_samples_leaf=1,           # Min samples per leaf
        min_samples_split=2,          # Min samples to split
        min_weight_fraction_leaf=0.0,
        random_state=42,              # Reproducible
        splitter='best'               # Best split
    )

    model.fit(X, y)
    score = model.score(X, y)

    return model, score


# -----------------------------
# Make Prediction
# -----------------------------
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

    return prediction[0], score


# -----------------------------
# Example Usage in Streamlit
# -----------------------------
if __name__ == "__main__":
    st.title("Diabetes Prediction App")

    # Load data
    df, X, y = load_data()

    st.write("Dataset preview:")
    st.dataframe(df.head())

    # User input
    st.subheader("Enter Patient Details:")
    HbA1c_level = st.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=5.5)
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.number_input("Glucose", min_value=50, max_value=300, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=70)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    BMI = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    input_features = [HbA1c_level, Pregnancies, Glucose, BloodPressure,
                      SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    if st.button("Predict"):
        prediction, acc = predict(X, y, input_features)
        if prediction == 1:
            st.error(f"⚠️ Patient is likely to have diabetes (Model Accuracy: {acc*100:.2f}%)")
        else:
            st.success(f"✅ Patient is unlikely to have diabetes (Model Accuracy: {acc*100:.2f}%)")
