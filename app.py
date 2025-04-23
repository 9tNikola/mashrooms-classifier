import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
# Page setup
st.set_page_config(
    page_title="🍄 Mushroom Edibility Classifier",
    page_icon="🍄",
    layout="centered"
)
# 🔧 Custom CSS
st.markdown("""
    <style>
        body {
            background-color: 
#f4f4f4;
        }
        .main {
            background-color: 
#ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h4 {
            color: 
#2d3436;
            font-family: 'Segoe UI', sans-serif;
        }
        .stSelectbox label {
            font-weight: 600;
            color: #333;
        }
        .stButton>button {
            background-color: 
#0984e3;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
        }
        .prediction-box {
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 10px;
            font-size: 1.3rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)
# ⬆️ Container
with st.container():
    st.title("🍄 Mushroom Edibility Classifier")
    st.markdown("##### 🚀 Predict if a mushroom is Edible or Poisonous based on its physical features.")
    st.write("This AI-powered model uses a trained Artificial Neural Network (ANN) to determine edibility.")
# 📦 Load model and encoders
model = load_model("mushroom_ann_model.keras")
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
# 🔠 Get feature names (excluding target)
feature_names = list(label_encoders.keys())[1:]
# ✍️ Input section
st.subheader("🔍 Input Mushroom Characteristics")
user_input = []
with st.form("input_form"):
    for feature in feature_names:
        options = labelencoders[feature].classes.tolist()
        selection = st.selectbox(f"🔸 {feature.replace('_', ' ').capitalize()}:", options)
        encoded_val = label_encoders[feature].transform([selection])[0]
        user_input.append(encoded_val)
    submitted = st.form_submit_button("🔮 Predict Edibility")
# 🧠 Predict
if submitted:
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0][0]
    st.subheader("🧾 Prediction Result")
    if prediction < 0.5:
        st.success("✅ The mushroom is EDIBLE")
        st.markdown(f"<div class='prediction-box' style='background-color: 
#dff9fb; color: 
#079992;'>🌿 Safe to consume — Confidence: {(1 - prediction) * 100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.error("❌ The mushroom is POISONOUS")
        st.markdown(f"<div class='prediction-box' style='background-color: 
#ffeaa7; color: 
#d63031;'>☠️ Not safe — Confidence: {prediction * 100:.2f}%</div>", unsafe_allow_html=True)
# 🧾 Footer
st.markdown("---")
st.markdown("<center><small>© 2025 Mushroom Classifier App | Developed by <b>Zia</b> with ❤️</small></center>", unsafe_allow_html=True)
