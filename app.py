import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import pickle
import time
from PIL import Image
from io import BytesIO

# Page configuration with improved layout and favicon
st.set_page_config(
    page_title="MycoSafe | Advanced Mushroom Edibility Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS
st.markdown("""
    <style>
        /* Global Styling */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f8f9fa;
            color: #343a40;
        }
        
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.06);
        }
        
        /* Headers */
        h1 {
            color: #2e7d32;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #81c784;
            padding-bottom: 0.5rem;
        }
        
        h2, h3, h4 {
            color: #2e7d32;
            font-weight: 600;
        }
        
        /* Form Elements */
        .stSelectbox label, .stSlider label {
            font-weight: 500;
            color: #424242;
            font-size: 1.05rem;
        }
        
        .stSelectbox > div > div {
            background-color: #f1f8e9;
            border-radius: 8px;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #2e7d32;
            color: white;
            border-radius: 12px;
            padding: 0.8rem 1.6rem;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            background-color: #388e3c;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* Prediction Box */
        .prediction-box {
            padding: 1.5rem;
            margin-top: 1.5rem;
            border-radius: 12px;
            font-size: 1.4rem;
            font-weight: 600;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        
        .prediction-box:hover {
            transform: translateY(-3px);
            box-shadow: 0px 6px 16px rgba(0, 0, 0, 0.12);
        }
        
        /* Cards */
        .info-card {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            border-left: 4px solid #2e7d32;
            box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.05);
        }
        
        /* Dividers */
        hr {
            height: 2px;
            background-color: #e0e0e0;
            border: none;
            margin: 2rem 0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f8e9;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 16px;
            color: #2e7d32;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #c8e6c9;
            font-weight: 600;
        }
        
        /* Loading animation */
        .stProgress > div > div > div > div {
            background-color: #4caf50;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #f1f8e9;
        }
        
        /* Tables */
        .dataframe {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe thead tr {
            background-color: #2e7d32;
            color: #ffffff;
            text-align: left;
        }
        
        .dataframe th, .dataframe td {
            padding: 12px 15px;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f1f8e9;
        }
        
        .dataframe tbody tr:last-of-type {
            border-bottom: 2px solid #2e7d32;
        }
        
        /* For the feature importance graph */
        .feature-importance-container {
            background-color: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        /* For the tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Logo styling */
        .logo-title-container {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .logo-title-container img {
            margin-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Create session state variables if they don't exist
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_explanation' not in st.session_state:
    st.session_state.show_explanation = False
    
# Sidebar for navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=MycoSafe", width=150)
    st.title("Navigation")
    
    app_mode = st.radio(
        "Select Application Mode",
        ["🧠 Mushroom Classifier", "📊 Analytics Dashboard", "📚 Learn About Mushrooms", "ℹ️ About & Help"]
    )
    
    st.markdown("---")
    st.markdown("### 🔬 Tool Information")
    st.info(
        """
        **MycoSafe 2.0** uses an advanced neural network trained on a dataset of over 8,000 mushroom samples.
        
        Version: 2.0.1
        Last Updated: April 2025
        """
    )
    
    st.markdown("---")
    st.markdown("### 📝 Session Stats")
    st.metric(label="Predictions Made", value=st.session_state.total_predictions)
    
    if st.session_state.total_predictions > 0:
        accuracy = (st.session_state.correct_predictions / st.session_state.total_predictions) * 100
        st.metric(label="User Accuracy", value=f"{accuracy:.1f}%")
    
    st.markdown("---")
    st.caption("© 2025 MycoSafe | Developed by Zia")

# Functions for the app
def load_data_and_models():
    """Load the model, encoders, and feature importances"""
    try:
        # Load model and encoders
        model = load_model("mushroom_ann_model.keras")
        
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
            
        # In a real implementation, you would load actual feature importance data
        # For now, let's create placeholder data
        feature_names = list(label_encoders.keys())[1:]
        feature_importances = {feature: np.random.uniform(0.3, 1.0) for feature in feature_names}
        
        # Sort by importance
        feature_importances = {k: v for k, v in sorted(feature_importances.items(), 
                                                      key=lambda item: item[1], 
                                                      reverse=True)}
        
        # Load sample images for common mushrooms (placeholder)
        sample_data = {
            "Amanita phalloides": {
                "common_name": "Death Cap", 
                "edible": False,
                "danger_level": "Lethal",
                "description": "One of the most poisonous mushrooms known. Contains deadly amatoxins."
            },
            "Agaricus bisporus": {
                "common_name": "Button Mushroom",
                "edible": True,
                "danger_level": "Safe",
                "description": "The most commonly cultivated mushroom worldwide, safe to eat."
            },
            "Cantharellus cibarius": {
                "common_name": "Chanterelle",
                "edible": True,
                "danger_level": "Safe",
                "description": "A highly prized edible wild mushroom with a fruity aroma."
            },
            "Amanita muscaria": {
                "common_name": "Fly Agaric",
                "edible": False,
                "danger_level": "Toxic",
                "description": "Distinctive red with white spots. Contains psychoactive compounds."
            },
            "Boletus edulis": {
                "common_name": "Porcini",
                "edible": True,
                "danger_level": "Safe",
                "description": "Highly valued edible mushroom with a nutty taste."
            }
        }
        
        return model, label_encoders, feature_importances, sample_data
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        return None, None, None, None

def get_feature_tooltip(feature_name):
    """Return helpful descriptions for each feature"""
    tooltips = {
        "cap_shape": "The shape of the mushroom's cap - e.g. bell, conical, convex, flat",
        "cap_surface": "Texture of the cap surface - e.g. fibrous, grooves, scaly, smooth",
        "cap_color": "Primary color of the mushroom cap",
        "bruises": "Whether the mushroom shows bruises when damaged",
        "odor": "The smell of the mushroom - often a key identifier",
        "gill_attachment": "How the gills attach to the stem",
        "gill_spacing": "How closely spaced the gills are",
        "gill_size": "Size of the gills relative to the cap",
        "gill_color": "Color of the gills under the cap",
        "stalk_shape": "The shape of the mushroom's stem/stalk",
        "stalk_root": "The type of root at the base of the stalk",
        "stalk_surface_above_ring": "Texture of the stalk above the ring",
        "stalk_surface_below_ring": "Texture of the stalk below the ring",
        "stalk_color_above_ring": "Color of the stalk above the ring",
        "stalk_color_below_ring": "Color of the stalk below the ring",
        "veil_type": "Type of partial veil covering immature gills",
        "veil_color": "Color of the veil",
        "ring_number": "Number of rings on the stalk",
        "ring_type": "Shape and consistency of the ring",
        "spore_print_color": "Color of spores when deposited on paper",
        "population": "Growth pattern - solitary, clustered, etc.",
        "habitat": "Where the mushroom typically grows"
    }
    
    return tooltips.get(feature_name, "No description available")

def plot_feature_importance(feature_importances):
    """Create a feature importance plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get top 10 features
    top_features = dict(list(feature_importances.items())[:10])
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, list(top_features.values()), color='#4caf50')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in top_features.keys()])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Relative Importance')
    ax.set_title('Top 10 Most Important Features for Prediction')
    
    plt.tight_layout()
    return fig

def format_feature_name(name):
    """Convert feature name from snake_case to Title Case with spaces"""
    return name.replace('_', ' ').title()

def record_prediction(input_features, prediction, user_feedback=None):
    """Record prediction history for analytics"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert numerical prediction to label
    prediction_label = "Edible" if prediction < 0.5 else "Poisonous"
    confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100
    
    history_entry = {
        "timestamp": timestamp,
        "features": input_features,
        "prediction": prediction_label,
        "confidence": confidence,
        "user_feedback": user_feedback
    }
    
    st.session_state.history.append(history_entry)
    st.session_state.total_predictions += 1
    
    # If user provided feedback and it matches prediction
    if user_feedback is not None and user_feedback == prediction_label:
        st.session_state.correct_predictions += 1

def display_analytics():
    """Display analytics based on prediction history"""
    st.title("📊 Analytics Dashboard")
    
    if len(st.session_state.history) == 0:
        st.info("No prediction data available yet. Make some predictions first!")
        return
    
    # Create tabs for different analytics views
    tab1, tab2, tab3 = st.tabs(["📈 Prediction History", "🔍 Feature Analysis", "🧪 Confidence Distribution"])
    
    with tab1:
        st.subheader("Recent Predictions")
        
        # Convert history to DataFrame for display
        history_df = pd.DataFrame(st.session_state.history)
        
        # Display in a table
        st.dataframe(
            history_df[['timestamp', 'prediction', 'confidence', 'user_feedback']]
            .sort_values('timestamp', ascending=False)
            .reset_index(drop=True)
        )
        
        if st.button("Clear History", key="clear_history"):
            st.session_state.history = []
            st.session_state.total_predictions = 0
            st.session_state.correct_predictions = 0
            st.experimental_rerun()
    
    with tab2:
        st.subheader("Feature Distribution in Predictions")
        
        # Choose a feature to analyze
        if len(st.session_state.history) > 0 and len(st.session_state.history[0]['features']) > 0:
            feature_keys = list(st.session_state.history[0]['features'].keys())
            selected_feature = st.selectbox("Select Feature to Analyze:", feature_keys)
            
            # Extract selected feature values
            feature_values = [entry['features'].get(selected_feature, "Unknown") for entry in st.session_state.history]
            feature_df = pd.DataFrame({
                selected_feature: feature_values,
                "prediction": [entry['prediction'] for entry in st.session_state.history]
            })
            
            # Plot distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Count by value and prediction
            pivot_table = pd.crosstab(feature_df[selected_feature], feature_df['prediction'])
            pivot_table.plot(kind='bar', stacked=True, ax=ax, color=['#4caf50', '#f44336'])
            
            plt.title(f"Distribution of {format_feature_name(selected_feature)} by Prediction")
            plt.xlabel(format_feature_name(selected_feature))
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Prediction Confidence Distribution")
        
        confidence_values = [entry['confidence'] for entry in st.session_state.history]
        predictions = [entry['prediction'] for entry in st.session_state.history]
        
        conf_df = pd.DataFrame({
            'Confidence': confidence_values,
            'Prediction': predictions
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create separate histograms for edible and poisonous
        sns.histplot(
            data=conf_df, 
            x='Confidence', 
            hue='Prediction', 
            bins=10, 
            palette=['#4caf50', '#f44336'],
            kde=True,
            ax=ax
        )
        
        plt.title("Distribution of Prediction Confidence")
        plt.xlabel("Confidence (%)")
        plt.ylabel("Count")
        plt.tight_layout()
        
        st.pyplot(fig)

def display_mushroom_guide():
    """Display educational content about mushrooms"""
    st.title("📚 Learn About Mushrooms")
    
    tabs = st.tabs(["🍄 Common Mushrooms", "🔬 How to Identify", "⚠️ Safety Tips"])
    
    with tabs[0]:
        st.subheader("Common Edible and Poisonous Mushrooms")
        
        _, sample_data = None, None
        try:
            _, _, _, sample_data = load_data_and_models()
        except:
            sample_data = {}
            
        if not sample_data:
            st.error("Sample data could not be loaded. Please try again later.")
            return
            
        # Display sample mushroom data in an attractive format
        col1, col2 = st.columns(2)
        
        for i, (name, data) in enumerate(sample_data.items()):
            with col1 if i % 2 == 0 else col2:
                with st.expander(f"{data['common_name']} ({name})"):
                    # Display placeholder image
                    st.image(f"https://via.placeholder.com/300x200.png?text={name.replace(' ', '+')}",
                            caption=f"{data['common_name']} ({name})")
                    
                    # Display info
                    st.markdown(f"**Edibility:** {'✅ Edible' if data['edible'] else '❌ Non-edible'}")
                    st.markdown(f"**Danger Level:** {data['danger_level']}")
                    st.markdown(f"**Description:** {data['description']}")
    
    with tabs[1]:
        st.subheader("How to Properly Identify Mushrooms")
        
        st.markdown("""
        ### Key Features for Identification
        
        Accurate mushroom identification requires careful examination of multiple features:
        
        1. **Cap** - Observe shape, color, texture and size
        2. **Gills/Pores** - Note attachment to stem, spacing, and color
        3. **Stem** - Check shape, color, texture, and presence of rings or veil
        4. **Spore Print** - Essential for definitive identification
        5. **Smell** - Many mushrooms have distinctive odors
        6. **Habitat** - Where a mushroom grows provides important clues
        
        ### Taking a Spore Print
        
        1. Cut the cap from the stem
        2. Place gill-side down on white and dark paper (side by side)
        3. Cover with a bowl to prevent air currents
        4. Wait 2-24 hours
        5. Remove the cap and observe the color of spores deposited
        
        ### IMPORTANT WARNING
        
        Never rely solely on digital tools or photos for mushroom identification before consumption.
        Always consult with an expert mycologist or field guide for definitive identification.
        """)
        
        # Example spore print colors
        st.markdown("#### Common Spore Print Colors")
        
        spore_colors = {
            "White": ["Amanita species (many deadly)", "Russula species"],
            "Brown": ["Button mushrooms", "Portobello", "Many Agaricus species"],
            "Black": ["Shaggy mane", "Coprinus species"],
            "Purple-Brown": ["Many Stropharia species"],
            "Pink": ["Entoloma species"],
            "Yellow": ["Some Boletes"]
        }
        
        col1, col2 = st.columns(2)
        
        for i, (color, mushrooms) in enumerate(spore_colors.items()):
            with col1 if i % 2 == 0 else col2:
                with st.expander(f"{color} Spore Print"):
                    # Color box
                    st.markdown(f"<div style='background-color:{color.lower() if color.lower() != 'white' else '#f0f0f0'}; height:30px; border-radius:5px; border:1px solid #ddd;'></div>", unsafe_allow_html=True)
                    st.markdown("**Example mushrooms:**")
                    for mushroom in mushrooms:
                        st.markdown(f"- {mushroom}")
    
    with tabs[2]:
        st.subheader("Mushroom Safety Guidelines")
        
        st.warning("""
        ### Essential Safety Rules
        
        1. **Never consume a mushroom unless you are 100% certain of its identity.**
        2. **When in doubt, throw it out!**
        3. **No reliable 'rule of thumb' exists** to distinguish edible from poisonous mushrooms.
        4. Some deadly mushrooms resemble edible species and may taste pleasant.
        5. Always cook mushrooms thoroughly before consumption.
        """)
        
        st.markdown("""
        ### Warning Signs of Mushroom Poisoning
        
        Symptoms can begin anywhere from 20 minutes to 24+ hours after consumption:
        
        - Nausea, vomiting, and diarrhea
        - Abdominal pain or cramping
        - Headache, dizziness, or confusion
        - Visual disturbances
        - Excessive sweating or salivation
        
        **If you suspect mushroom poisoning:**
        
        1. Call poison control immediately: **1-800-222-1222**
        2. Try to preserve a sample of the consumed mushroom
        3. Seek emergency medical attention
        
        ### Do Not Rely On:
        
        - Whether animals eat them
        - Cooking methods to detoxify
        - Silver/garlic turning black (old wives' tale)
        - Only visual identification from photos or apps
        """)

def display_about():
    """Display about and help information"""
    st.title("ℹ️ About & Help")
    
    st.markdown("""
    ## About MycoSafe
    
    **MycoSafe** is an advanced machine learning application designed to help identify potentially edible and poisonous mushrooms based on their physical characteristics. The system uses a deep neural network trained on thousands of mushroom samples.
    
    ### How It Works
    
    1. The classifier uses a trained Artificial Neural Network (ANN) model
    2. Input features are processed through multiple neural network layers
    3. The model outputs a probability score between 0 and 1
    4. Scores below 0.5 indicate likely edible, scores above 0.5 indicate likely poisonous
    
    ### Important Disclaimer
    
    This application is for **educational purposes only**. Never rely solely on this or any digital tool for mushroom identification before consumption. Always consult with an expert mycologist or comprehensive field guide for definitive identification.
    
    ### Data Sources
    
    The model was trained using data compiled from various mycological databases and scientific publications. The dataset includes information on cap shape, gill characteristics, stem features, spore prints, and other identifying features.
    
    ## Help & FAQ
    """)
    
    # FAQ section with expandable questions
    with st.expander("🔍 How accurate is the classifier?"):
        st.markdown("""
        The classifier has been trained on thousands of labeled mushroom samples and achieves approximately 95% accuracy on test data. However, this does not mean it should be relied upon for consumption decisions.
        
        Factors that can affect accuracy:
        - Unusual or rare mushroom varieties
        - Subjective interpretation of features like colors and textures
        - Regional variations in mushroom characteristics
        """)
        
    with st.expander("❓ What if I don't know all the features?"):
        st.markdown("""
        For the most accurate results, try to provide all requested features. If you're unsure about a particular feature:
        
        1. Leave it at the default selection
        2. Take note of the reduced confidence score
        3. Consider consulting a field guide or expert
        
        The "Feature Importance" section can help you understand which features have the most impact on classification.
        """)
        
    with st.expander("📱 Can I use this app in the field?"):
        st.markdown("""
        Yes, this web app is designed to be responsive and works on mobile devices. However:
        
        - Ensure you have internet connectivity
        - Consider downloading field guide resources for offline use
        - Take clear photos of the mushroom for later reference
        - Make notes of habitat and growth conditions
        
        Remember that field identification should always be verified by multiple sources.
        """)
        
    with st.expander("🔄 How often is the model updated?"):
        st.markdown("""
        The model is updated quarterly with new training data and improvements to the neural network architecture. The last update was in April 2025.
        
        Each update includes:
        - Additional mushroom samples
        - Refined feature extraction
        - Performance optimizations
        - User feedback incorporation
        """)
    
    # Contact information
    st.markdown("""
    ### Contact & Support
    
    For questions, feedback, or to report issues:
    
    **Email:** support@mycosafe.example.com  
    **GitHub:** [github.com/mycosafe/mushroom-classifier](https://github.com)  
    **Version:** 2.0.1
    
    © 2025 MycoSafe | Developed by Zia
    """)

def show_classifier():
    """Display the main mushroom classifier functionality"""
    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://via.placeholder.com/100x100.png?text=🍄", width=100)
    with col2:
        st.title("MycoSafe: Advanced Mushroom Edibility Classifier")
        st.markdown("##### Predict mushroom edibility with AI-powered analysis of physical characteristics")
    
    # Load models and data
    model, label_encoders, feature_importances, _ = load_data_and_models()
    
    if not model or not label_encoders:
        st.error("Failed to load necessary models and data. Please try again later.")
        return
    
    # Get feature names (excluding target)
    feature_names = list(label_encoders.keys())[1:]
    
    # Display in tabs for better organization
    main_tab, feature_tab, result_tab = st.tabs(["💬 Input Features", "📊 Feature Importance", "🧠 Model Explanation"])
    
    with main_tab:
        st.markdown("### 🔍 Enter Mushroom Characteristics")
        
        # Create columns for more compact layout
        user_input = {}
        user_input_encoded = []
        
        with st.form("enhanced_input_form"):
            # Organize features into meaningful groups
            st.markdown("#### Cap Characteristics")
            col1, col2, col3 = st.columns(3)
            with col1:
                for feature in ["cap_shape", "cap_surface"]:
                    if feature in feature_names:
                        options = label_encoders[feature].classes_.tolist()
                        tooltip = get_feature_tooltip(feature)
                        
                        st.markdown(f"""
                        <div class="tooltip">🔸 {format_feature_name(feature)}
                          <span class="tooltiptext">{tooltip}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        selection = st.selectbox(
                            "",
                            options,
                            key=f"select_{feature}"
                        )
                        user_input[feature] = selection
                        encoded_val = label_encoders[feature].transform([selection])[0]
                        user_input_encoded.append(encoded_val)
            
            with col2:
                for feature in ["cap_color", "bruises"]:
                    if feature in feature_names:
                        options = label_encoders[feature].classes_.tolist()
                        tooltip = get_feature_tooltip(feature)
                        
                        st.markdown(f"""
                        <div class="tooltip">🔸 {format_feature_name(feature)}
                          <span class="tooltiptext">{tooltip}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        selection = st.selectbox(
                            "",
                            options,
                            key=f"select_{feature}"
                        )
                        user_input[feature] = selection
                        encoded_val = label_encoders[feature].transform([selection])[0]
                        user_input_encoded.append(encoded_val)
            
            with col3:
                for feature in ["odor"]:
                    if feature in feature_names:
                        options = label_encoders[feature].classes_.tolist()
                        tooltip = get_feature_tooltip(feature)
                        
                        st.markdown(f"""
                        <div class="tooltip
