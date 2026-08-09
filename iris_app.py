import streamlit as st
import pickle
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Title and description
st.title("🌺 Iris Flower Species Classifier")
st.write("This app uses a pre-trained Machine Learning model (`iris_model.pkl`) to predict the species of an Iris flower based on its feature values.")

# Load the iris target names for human-readable output
iris_data = load_iris()
target_names = iris_data.target_names  # ['setosa', 'versicolor', 'virginica']

# Load pre-trained model

# def load_model():
#     with open('iris_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model
@st.cache_resource
def load_model():
    model = joblib.load("iris_model.pkl")
    return model

try:
    model = load_model()
    st.sidebar.header("Input Flower Features")

    # Inputs corresponding to the 4 features of the Iris dataset:
    # Feature 0: Sepal length (numeric)
    # Feature 1: Sepal width (numeric)
    # Feature 2: Petal length (numeric)
    # Feature 3: Petal width (categorical/binned or numeric in pipeline)
    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(iris_data.data[:, 0].min()), float      (iris_data.data[:, 0].max()), float(iris_data.data[:, 0].mean()))
    sepal_width = st.sidebar.slider("Sepal Width (cm)", float(iris_data.data[:, 1].min()), float(iris_data.data[:, 1].max()), float(iris_data.data[:, 1].mean()))
    petal_length = st.sidebar.slider("Petal Length (cm)", float(iris_data.data[:, 2].min()), float(iris_data.data[:, 2].max()), float(iris_data.data[:, 2].mean()))
    petal_width = st.sidebar.slider("Petal Width (cm)", float(iris_data.data[:, 3].min()), float(iris_data.data[:, 3].max()), float(iris_data.data[:, 3].mean()))

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    st.subheader("Input Values")
    st.write({
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    })

    if st.button("Predict Species"):
        prediction = model.predict(input_data)[0]
        predicted_species = target_names[prediction]

        st.success(f"**Predicted Species:** {predicted_species.capitalize()}")

except FileNotFoundError:
    st.error("⚠️ `iris_model.pkl` file not found. Please ensure the model file is saved in the working directory.")
