import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Page config
st.set_page_config(
    page_title="🍄 Mushroom Edibility Classifier",
    page_icon="🍄",
    layout="centered"
)

# Load model and encoders
model = load_model("mushroom_ann_model.keras")
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Only use important fields
important_features = [
    "cap_shape",
    "cap_color",
    "gill_size",
    "gill_color",
    "ring_type",
    "spore_print_color",
    "habitat"
]

# Custom CSS for styling
st.markdown("""
    <style>
        .stSelectbox label {
            font-weight: 600;
            color: #333;
        }
        .stButton>button {
            background-color: #0abde3;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .prediction-box {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.3rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🍄 Mushroom Edibility Classifier")
st.markdown("### 🚀 Find out if your mushroom is **safe or poisonous** using AI")

st.write("Enter the following key characteristics of the mushroom:")

# Form for user input
user_input = []
with st.form("input_form"):
    for feature in important_features:
        options = label_encoders[feature].classes_.tolist()
        selection = st.selectbox(f"🔹 {feature.replace('_', ' ').capitalize()}:", options)
        encoded_val = label_encoders[feature].transform([selection])[0]
        user_input.append(encoded_val)

    submitted = st.form_submit_button("🔮 Predict Edibility")

# Prediction
if submitted:
    input_array = np.zeros((1, len(label_encoders) - 1))  # Total features minus target
    for idx, feature in enumerate(important_features):
        feature_index = list(label_encoders.keys()).index(feature) - 1
        input_array[0][feature_index] = user_input[idx]

    prediction = model.predict(input_array)[0][0]

    st.subheader("🧾 Prediction Result")
    if prediction < 0.5:
        st.success("✅ This mushroom is **EDIBLE**")
        st.markdown(f"<div class='prediction-box' style='background-color: #dff9fb; color: #079992;'>Confidence: {(1 - prediction) * 100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.error("☠️ This mushroom is **POISONOUS**")
        st.markdown(f"<div class='prediction-box' style='background-color: #ffeaa7; color: #d63031;'>Confidence: {prediction * 100:.2f}%</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center><small>App by Zia | Powered by Streamlit & TensorFlow</small></center>", unsafe_allow_html=True)
