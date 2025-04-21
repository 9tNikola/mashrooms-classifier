import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Page config
st.set_page_config(page_title="🍄 Mushroom Edibility Predictor", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f5f0;
        }
        h1 {
            color: #4d2600;
            font-family: 'Trebuchet MS', sans-serif;
        }
        .stSelectbox label {
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .result {
            font-size: 1.5rem;
            font-weight: bold;
            color: #d63333;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("🍄 Mushroom Edibility Classifier")
st.markdown("#### Find out if a mushroom is **edible** or **poisonous** based on its characteristics.")
st.write("Enter the properties below to get a real-time prediction using a trained neural network model.")

# Load the trained model
model = load_model("mushroom_ann_model.keras")

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Get features (excluding target column)
feature_names = list(label_encoders.keys())[1:]

# Input form
user_input = []
with st.form("input_form"):
    for feature in feature_names:
        options = label_encoders[feature].classes_.tolist()
        selection = st.selectbox(f"Select {feature.replace('_', ' ').capitalize()}:", options)
        encoded_val = label_encoders[feature].transform([selection])[0]
        user_input.append(encoded_val)

    submitted = st.form_submit_button("Predict Mushroom Edibility")

# Predict
if submitted:
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0][0]

    if prediction < 0.5:
        st.success("🍽️ This mushroom is **EDIBLE**! Safe to eat (but still be cautious in real life!)")
        st.markdown("<div class='result' style='color: green;'>✅ Neural Network Confidence: {:.2f}%</div>".format((1 - prediction) * 100), unsafe_allow_html=True)
    else:
        st.error("☠️ This mushroom is **POISONOUS**! Not safe to consume.")
        st.markdown("<div class='result'>⚠️ Neural Network Confidence: {:.2f}%</div>".format(prediction * 100), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using TensorFlow & Streamlit | Project by Zia")
