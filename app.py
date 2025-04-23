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

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3, h4 {
            color: #2d3436;
            font-family: 'Segoe UI', sans-serif;
        }
        .stSelectbox label, .stRadio label {
            font-weight: 600;
            color: #333;
        }
        .stButton>button {
            background-color: #0984e3;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #0767b3;
            transform: scale(1.02);
        }
        .prediction-box {
            padding: 1.5rem;
            margin-top: 1rem;
            border-radius: 10px;
            font-size: 1.3rem;
            font-weight: 600;
            text-align: center;
        }
        .feature-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Container
with st.container():
    st.title("🍄 Mushroom Edibility Classifier")
    st.markdown("##### 🚀 Predict if a mushroom is edible or poisonous based on its physical characteristics")
    st.write("This AI-powered model uses a trained Artificial Neural Network (ANN) to determine mushroom edibility with high accuracy.")
    st.markdown("---")

# Load model and encoders
@st.cache_resource
def load_assets():
    model = load_model("mushroom_ann_model.keras")
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    return model, label_encoders

try:
    model, label_encoders = load_assets()
    feature_names = list(label_encoders.keys())[1:]  # Exclude target variable
except Exception as e:
    st.error(f"Error loading model or encoders: {str(e)}")
    st.stop()

# Mushroom feature descriptions
feature_descriptions = {
    'cap-shape': "The shape of the mushroom cap (bell, conical, flat, etc.)",
    'cap-surface': "Texture of the cap surface (fibrous, grooves, scaly, smooth)",
    'cap-color': "Color of the mushroom cap",
    'bruises': "Whether the mushroom bruises or changes color when pressed",
    'odor': "Distinct smell of the mushroom",
    'gill-attachment': "How gills attach to the stem",
    'gill-spacing': "Spacing between the gills",
    'gill-size': "Size of the gills",
    'gill-color': "Color of the gills",
    'stalk-shape': "Shape of the stalk (tapering, enlarging, etc.)",
    'stalk-root': "Root structure of the stalk",
    'stalk-surface-above-ring': "Texture of stalk surface above the ring",
    'stalk-surface-below-ring': "Texture of stalk surface below the ring",
    'stalk-color-above-ring': "Color of stalk above the ring",
    'stalk-color-below-ring': "Color of stalk below the ring",
    'veil-type': "Type of veil (partial membrane covering)",
    'veil-color': "Color of the veil",
    'ring-number': "Number of rings on the stalk",
    'ring-type': "Type of ring",
    'spore-print-color': "Color of the spore print",
    'population': "How the mushrooms grow (clustered, scattered, etc.)",
    'habitat': "Where the mushroom typically grows"
}

# Input section
st.subheader("🔍 Input Mushroom Characteristics")
st.markdown("Select the physical features of the mushroom you want to analyze:")

user_input = []
with st.form("input_form"):
    cols = st.columns(2)
    col_idx = 0
    
    for i, feature in enumerate(feature_names):
        with cols[col_idx]:
            with st.expander(f"**{feature.replace('-', ' ').title()}**"):
                st.caption(feature_descriptions.get(feature, "No description available"))
                options = label_encoders[feature].classes_.tolist()
                selection = st.selectbox(
                    f"Select {feature.replace('-', ' ')}:",
                    options,
                    key=feature,
                    help=f"Select the {feature.replace('-', ' ')} characteristic"
                )
                encoded_val = label_encoders[feature].transform([selection])[0]
                user_input.append(encoded_val)
        
        col_idx = 1 if col_idx == 0 else 0
    
    submitted = st.form_submit_button("🔮 Predict Edibility", use_container_width=True)

# Prediction
if submitted:
    try:
        input_array = np.array(user_input).reshape(1, -1)
        prediction = model.predict(input_array)[0][0]
        
        st.markdown("---")
        st.subheader("🧾 Prediction Result")
        
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.success("✅ The mushroom is **EDIBLE**")
            st.markdown(
                f"<div class='prediction-box' style='background-color: #e3f9e5; color: #1b5e20;'>"
                f"🌿 Safe to consume — Confidence: {confidence:.2f}%</div>", 
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            confidence = prediction * 100
            st.error("❌ The mushroom is **POISONOUS**")
            st.markdown(
                f"<div class='prediction-box' style='background-color: #ffebee; color: #c62828;'>"
                f"☠️ Dangerous — Do not consume! Confidence: {confidence:.2f}%</div>", 
                unsafe_allow_html=True
            )
            
        # Add some additional information
        st.markdown("---")
        with st.expander("ℹ️ About the prediction"):
            st.write("""
            - **Confidence score** represents how certain the model is about its prediction (0-100%)
            - **Edible** means the mushroom is likely safe for consumption
            - **Poisonous** means the mushroom may be harmful or deadly if ingested
            - Always consult with an expert mycologist before consuming wild mushrooms
            """)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <small>© 2025 Mushroom Classifier App | Developed by <b>Zia</b> with ❤️</small><br>
        <small>Disclaimer: This tool is for educational purposes only. Always verify with an expert before consumption.</small>
    </div>
""", unsafe_allow_html=True)
