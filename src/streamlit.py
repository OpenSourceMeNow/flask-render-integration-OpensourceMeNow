import streamlit as st
from joblib import load
import traceback

# Load the model
try:
    model = load("../models/decision_tree_classifier_default_42.sav")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.error(traceback.format_exc())
    model = None

# Define the class dictionary
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

# Set the page title
st.set_page_config(page_title="Iris Prediction", page_icon=":flower:")

# Create the Streamlit app
def main():
    st.title("Iris Prediction")

    # Create input fields for the feature values
    val1 = st.number_input("Petal width", value=0.0, step=0.1)
    val2 = st.number_input("Petal length", value=0.0, step=0.1)
    val3 = st.number_input("Sepal width", value=0.0, step=0.1)
    val4 = st.number_input("Sepal length", value=0.0, step=0.1)

    # Create a button to trigger the prediction
    if st.button("Predict"):
        try:
            data = [[val1, val2, val3, val4]]
            prediction = str(model.predict(data)[0])
            pred_class = class_dict[prediction]
            st.success(f"Prediction: {pred_class}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()