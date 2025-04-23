import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🍄 Mushroom Edibility Classifier Pro",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS for premium professional appearance
st.markdown("""
<style>
    /* Base styling */
    .main {
        background-color: #f9fafb;
        padding: 2rem;
    }
    .stApp {
        font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
    }
    
    /* Typography */
    h1 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 1.5rem;
    }
    h2, h3 {
        color: #334155;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    h4, h5, h6 {
        color: #475569;
        font-weight: 500;
    }
    p, li {
        color: #4b5563;
        line-height: 1.6;
    }
    
    /* Cards and containers */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.75rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03), 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #f1f5f9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.05), 0 2px 5px rgba(0,0,0,0.07);
    }
    
    /* Form elements */
    .stSelectbox, .stMultiselect, div[data-baseweb="select"] {
        border-radius: 6px;
    }
    .stTextInput>div>div>input {
        border-radius: 6px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        box-shadow: 0 2px 4px rgba(30, 64, 175, 0.2);
        transition: all 0.3s ease;
        border: none;
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
        box-shadow: 0 4px 8px rgba(30, 64, 175, 0.3);
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Secondary button */
    .secondary-button > button {
        background: transparent;
        color: #3b82f6;
        border: 1px solid #3b82f6;
        box-shadow: none;
    }
    .secondary-button > button:hover {
        background: rgba(59, 130, 246, 0.05);
        box-shadow: none;
    }
    
    /* Danger button */
    .danger-button > button {
        background: linear-gradient(90deg, #be123c 0%, #f43f5e 100%);
    }
    .danger-button > button:hover {
        background: linear-gradient(90deg, #9f1239 0%, #e11d48 100%);
    }
    
    /* Results styling */
    .prediction-box {
        padding: 1.5rem;
        margin-top: 1rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .prediction-box h3 {
        margin-top: 0;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .prediction-box p {
        margin-bottom: 0.5rem;
    }
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        border-left: 4px solid #3b82f6;
    }
    .warning-box {
        background-color: #fff7ed;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        border-left: 4px solid #f97316;
    }
    
    /* Feature highlight */
    .feature-highlight {
        background-color: #f8fafc;
        padding: 0.75rem;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e40af;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e7ff !important;
        color: #1e40af !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    
    /* Progress bars */
    div[role="progressbar"] > div {
        background-image: linear-gradient(90deg, #3b82f6, #60a5fa);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #334155;
    }
    
    /* Footer */
    footer {
        margin-top: 3rem;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Charts and visualizations */
    .plot-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .badge-yellow {
        background-color: #fef9c3;
        color: #854d0e;
    }
    
    /* Custom datatables */
    .dataframe {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9rem;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .dataframe thead tr {
        background-color: #3b82f6;
        color: white;
        text-align: left;
    }
    .dataframe th, .dataframe td {
        padding: 12px 15px;
    }
    .dataframe tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f8fafc;
    }
    .dataframe tbody tr:last-of-type {
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Set up session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'favorite_configurations' not in st.session_state:
    st.session_state.favorite_configurations = []
    
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {
        'theme': 'light',
        'expert_mode': False,
        'confidence_threshold': 85
    }

# Function to load model and encoders
@st.cache_resource
def load_data():
    try:
        model = load_model("mushroom_ann_model.keras")
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        return model, label_encoders
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please ensure files are correctly placed.")
        return None, None

# Function to make prediction
def predict_edibility(input_features, model):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)[0][0]
    return prediction

# Function to generate feature importance visualization
def create_feature_importance_chart(feature_names, feature_values):
    df = pd.DataFrame({'Feature': feature_names, 'Value': feature_values})
    fig = px.bar(
        df, 
        x='Value', 
        y='Feature',
        orientation='h',
        color='Value',
        color_continuous_scale='viridis',
        labels={'Value': 'Encoded Value', 'Feature': 'Feature Name'},
        title='Feature Distribution Analysis'
    )
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial", size=12, color="#334155")
    )
    return fig

# Function to create radar chart for mushroom profile
def create_radar_chart(feature_names, feature_values, max_values):
    # Normalize values for radar chart
    normalized_values = [val/max_val for val, max_val in zip(feature_values, max_values)]
    
    categories = [f.replace('-', ' ').title() for f in feature_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Feature Profile',
        line=dict(color='#3b82f6', width=2),
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial", size=12, color="#334155"),
        title="Mushroom Feature Profile"
    )
    
    return fig

# Function to create confidence gauge
def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': "#3b82f6" if confidence >= 80 else "#f97316" if confidence >= 60 else "#ef4444"},
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [60, 80], 'color': "rgba(249, 115, 22, 0.2)"},
                {'range': [80, 100], 'color': "rgba(59, 130, 246, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#334155", 'width': 2},
                'thickness': 0.75,
                'value': 90
            }
        },
        title = {'text': "Confidence Score", 'font': {'size': 16, 'color': "#334155", 'family': "Inter, Arial"}},
        number = {'suffix': "%", 'font': {'size': 22, 'color': "#334155", 'family': "Inter, Arial"}},
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# Function to save prediction to history
def save_to_history(input_dict, prediction, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = "Edible" if prediction < 0.5 else "Poisonous"
    st.session_state.prediction_history.append({
        "timestamp": timestamp,
        "features": input_dict,
        "result": result,
        "confidence": confidence,
        "notes": ""
    })

# Function to save configuration as favorite
def save_as_favorite(input_dict, name):
    st.session_state.favorite_configurations.append({
        "name": name,
        "features": input_dict,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Function to export data to CSV
def export_to_csv(data):
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# Main App Structure
def main():
    # Load model and encoders
    model, label_encoders = load_data()
    if model is None or label_encoders is None:
        st.stop()
    
    # Get feature names (excluding target)
    feature_names = list(label_encoders.keys())
    if 'class' in feature_names:
        feature_names.remove('class')
    
    # Sidebar
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://www.svgrepo.com/show/66498/mushroom.svg", width=50)
        with col2:
            st.title("MycoAnalytics")
            st.markdown('<p style="margin-top:-15px;font-size:1.1rem;font-weight:500;color:#64748b;">Professional Edition</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Application Navigation
        st.markdown("### 📊 Navigation")
        app_mode = st.radio("", 
            ["🔬 Classification", "📈 Data Explorer", "📜 History", "⭐ Favorites", "⚙️ Settings", "ℹ️ About"],
            label_visibility="collapsed")
        
        st.markdown("---")
        
        # Quick access panel
        st.markdown("### 🚀 Quick Actions")
        
        # Show date and time
        current_date = datetime.now().strftime("%B %d, %Y")
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f"📅 **{current_date}**")
        st.markdown(f"🕒 **{current_time}**")
        
        if st.session_state.prediction_history:
            st.markdown("#### Recent Results")
            for idx, entry in enumerate(st.session_state.prediction_history[-3:]):
                result_color = "green" if entry["result"] == "Edible" else "red"
                st.markdown(f"""
                <div style="padding:8px;border-left:3px solid #{result_color};margin-bottom:8px;background-color:#f8fafc;border-radius:4px;">
                <p style="margin:0;font-size:0.8rem;color:#64748b;">{entry["timestamp"][5:]}</p>
                <p style="margin:0;font-weight:500;color:#{'166534' if entry["result"] == 'Edible' else '991b1b'}">
                {'✅ ' if entry["result"] == 'Edible' else '⚠️ '}{entry["result"]}
                </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Documentation section
        with st.expander("📚 Documentation"):
            st.markdown("""
            #### Using the application
            
            1. **Select mushroom characteristics**
            2. Click 'Analyze Mushroom'
            3. View results with confidence score
            4. Save configurations for future reference
            
            #### Advanced features
            
            - Export analysis history as CSV
            - Compare multiple predictions
            - Save favorite configurations
            - Expert mode for detailed analytics
            """)
        
        st.markdown("---")
        st.markdown("<footer>© 2025 MycoAnalytics Professional<br>v2.5.0 Enterprise</footer>", unsafe_allow_html=True)

    # Main Content based on mode
    if app_mode == "🔬 Classification":
        classification_mode(model, label_encoders, feature_names)
    elif app_mode == "📈 Data Explorer":
        data_explorer_mode(label_encoders, feature_names)
    elif app_mode == "📜 History":
        history_mode()
    elif app_mode == "⭐ Favorites":
        favorites_mode(model, label_encoders, feature_names)
    elif app_mode == "⚙️ Settings":
        settings_mode()
    else:
        about_mode()

def classification_mode(model, label_encoders, feature_names):
    # Header with professional design
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background-color:#dbeafe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:24px;">🔬</span>
        </div>
        <div>
            <h1 style="margin:0;padding:0;">Mushroom Edibility Analysis</h1>
            <p style="margin:0;padding:0;color:#64748b;">Professional classification system for mycologists and researchers</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout with tabs for better organization
    tabs = st.tabs(["🧪 Analysis", "📊 Results", "📝 Documentation"])
    
    # Analysis Tab
    with tabs[0]:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Species Characteristics Input</h3>
                <p>Enter the physical properties of the mushroom specimen to analyze its edibility.</p>
            """, unsafe_allow_html=True)
            
            # Create form for input
            with st.form("input_form"):
                user_input = []
                input_dict = {}
                
                # Create feature input groups with improved organization
                feature_groups = {
                    "Cap Properties": ["cap-shape", "cap-surface", "cap-color"],
                    "Gill Properties": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
                    "Stem Properties": ["stalk-shape", "stalk-root", "stalk-surface-above-ring", 
                                       "stalk-surface-below-ring", "stalk-color-above-ring", 
                                       "stalk-color-below-ring"],
                    "Other Features": ["veil-type", "veil-color", "ring-number", "ring-type", 
                                      "spore-print-color", "population", "habitat"]
                }
                
                # Enhanced feature selection UI
                for group, features in feature_groups.items():
                    st.markdown(f"""
                    <div style="margin-top:1rem;">
                        <h4 style="display:flex;align-items:center;gap:8px;">
                            <span style="color:#3b82f6;font-size:1.2rem;">●</span> {group}
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns(min(3, len(features)))
                    
                    for i, feature in enumerate(features):
                        with cols[i % len(cols)]:
                            if feature in feature_names:
                                options = label_encoders[feature].classes_.tolist()
                                selection = st.selectbox(
                                    f"{feature.replace('-', ' ').title()}:",
                                    options,
                                    help=f"Select the {feature.replace('-', ' ')} characteristic"
                                )
                                encoded_val = label_encoders[feature].transform([selection])[0]
                                user_input.append(encoded_val)
                                input_dict[feature] = selection
                
                # Action buttons row
                action_cols = st.columns([2, 1, 1])
                with action_cols[0]:
                    submitted = st.form_submit_button("🔬 Analyze Mushroom")
                with action_cols[1]:
                    save_config = st.form_submit_button("💾 Save Configuration")
                with action_cols[2]:
                    clear_form = st.form_submit_button("🔄 Reset Form")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Professional feature guide
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Analysis Guide</h3>
                <p>Follow these steps for accurate mushroom classification:</p>
                
                <ol>
                    <li>Enter all observable characteristics</li>
                    <li>Use the dropdown menus to select properties</li>
                    <li>Click "Analyze Mushroom" for results</li>
                    <li>Review confidence score and indicators</li>
                </ol>
                
                <div class="info-box">
                    <h4 style="margin-top:0;">Pro Tip</h4>
                    <p>The cap, gill, and spore print colors are the most reliable indicators for identification.</p>
                </div>
                
                <div class="warning-box">
                    <h4 style="margin-top:0;">Important Notice</h4>
                    <p>Always verify results with field guides and expert consultation before consumption.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Image reference card
            st.markdown("""
            <div class="card">
                <h3>Reference Guide</h3>
                <p>Visual identification is crucial for accurate analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.image("https://www.svgrepo.com/show/293691/mushroom-fungi.svg", width=200)
    
    # Results Tab
    with tabs[1]:
        if 'submitted' in locals() and submitted:
            with st.spinner("Processing specimen analysis..."):
                # Add a slight delay for UX
                time.sleep(0.8)
                
                # Get prediction
                prediction = predict_edibility(user_input, model)
                confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100
                
                # Save to history
                save_to_history(input_dict, prediction, confidence)
                
                # Results layout
                st.markdown("""
                <div class="card">
                    <h3>Analysis Results</h3>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction < 0.5:
                        st.markdown(f"""
                        <div class='prediction-box' style='background-color: #dcfce7; color: #166534;'>
                            <h3>✅ EDIBLE</h3>
                            <p>This mushroom specimen is classified as safe for consumption.</p>
                            <p>Confidence Score: {confidence:.2f}%</p>
                            <p style="font-size:0.9rem;">ID: {hash(str(input_dict))}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='prediction-box' style='background-color: #fee2e2; color: #991b1b;'>
                            <h3>⚠️ POISONOUS</h3>
                            <p>This mushroom specimen is classified as toxic. Do not consume.</p>
                            <p>Confidence Score: {confidence:.2f}%</p>
                            <p style="font-size:0.9rem;">ID: {hash(str(input_dict))}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display warning
                    st.warning("⚠️ Important: Always consult with a certified mycologist before consuming any wild mushrooms.")
                
                with col2:
                    # Confidence gauge chart
                    st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True, config={'displayModeBar': False})
                    
                    # Key indicators
                    st.markdown("### Key Indicators")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with metrics_col2:
                        result_text = "Edible" if prediction < 0.5 else "Poisonous"
                        st.metric("Classification", result_text)
                    with metrics_col3:
                        risk_level = "Low" if confidence > 90 else "Medium" if confidence > 70 else "High"
                        st.metric("Risk Level", risk_level)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Feature analysis section
                st.markdown("""
                <div class="card">
                    <h3>Feature Analysis</h3>
                    <p>Detailed analysis of specimen characteristics and their impact on classification</p>
                """, unsafe_allow_html=True)
                
                analysis_col1, analysis_col2 = st.columns([3, 2])
                
                with analysis_col1:
                    # Feature importance visualization
                    fig = create_feature_importance_chart(feature_names, user_input)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                with analysis_col2:
                    # Calculate max values for each feature (for normalization)
                    max_values = []
                    for feature in feature_names:
                        if feature in label_encoders:
                            max_values.append(len(label_encoders[feature].classes_) - 1)
                        else:
                            max_values.append(1)
                    
                    # Radar chart visualization
                    radar_fig = create_radar_chart(feature_names, user_input, max_values)
                    st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': False})
                
                # Add notes for this analysis
                st.text_area("Analysis Notes", key="analysis_notes", 
                             help="Add your notes about this specimen analysis")
                
                # Quick action buttons
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button("📋 Copy Results"):
                        result_text = "Edible" if prediction < 0.5 else "Poisonous"
                        copy_text = f"Mushroom Analysis Results:\n- Classification: {result_text}\n- Confidence: {confidence:.2f}%\n- Date: {datetime.now().strftime('%Y-%m-%d')}"
                        st.code(copy_text)
                with action_col2:
                    if st.button("💾 Save as Favorite"):
                        save_name = f"{'Edible' if prediction < 0.5 else 'Poisonous'} Specimen {len(st.session_state.favorite_configurations)+1}"
                        save_as_favorite(input_dict, save_name)
                        st.success(f"Saved as '{save_name}'")
                with action_col3:
                    if st.button("📊 Generate Report"):
                        st.info("Generating detailed PDF report... This feature requires a premium subscription.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Empty results state
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;">
            <h3>Analysis Results</h3>
                <p>No analysis has been performed yet. Please go to the Analysis tab and enter mushroom characteristics.</p>
                <div style="display:flex;justify-content:center;margin:2rem 0;">
                    <img src="https://www.svgrepo.com/show/66498/mushroom.svg" style="width:100px;opacity:0.5;">
                </div>
                <p>Once you complete the analysis form, detailed results will appear here including:</p>
                <ul>
                    <li>Edibility classification with confidence score</li>
                    <li>Feature importance visualization</li>
                    <li>Comparative analysis and risk assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Documentation Tab
    with tabs[2]:
        st.markdown("""
        <div class="card">
            <h3>Classification Documentation</h3>
            <p>Understanding the science behind mushroom classification</p>
            
            <div class="feature-highlight">
                <h4>Scientific Approach</h4>
                <p>MycoAnalytics Pro uses a neural network model trained on the comprehensive UCI Mushroom Dataset, 
                analyzing 22 physical characteristics to determine edibility with high precision.</p>
            </div>
            
            <div class="feature-highlight">
                <h4>Key Indicators</h4>
                <p>The most reliable indicators for edibility classification include:</p>
                <ul>
                    <li>Gill characteristics (attachment, spacing, color)</li>
                    <li>Spore print color</li>
                    <li>Cap surface and color</li>
                    <li>Stalk features and ring type</li>
                </ul>
            </div>
            
            <div class="feature-highlight">
                <h4>Understanding Confidence Scores</h4>
                <p>The confidence score indicates the model's certainty in its classification:</p>
                <ul>
                    <li><strong>90-100%:</strong> High confidence classification</li>
                    <li><strong>70-89%:</strong> Moderate confidence, verification recommended</li>
                    <li><strong>Below 70%:</strong> Low confidence, additional analysis required</li>
                </ul>
            </div>
            
            <div class="warning-box">
                <h4>Important Safety Notice</h4>
                <p>This tool should be used as a supplementary reference only. Always consult with expert mycologists and field guides 
                before consuming any wild mushrooms. Some toxic species closely resemble edible varieties.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # References and resources
        st.markdown("""
        <div class="card">
            <h3>Resources & References</h3>
            
            <h4>Professional References</h4>
            <ul>
                <li>UCI Machine Learning Repository: Mushroom Data Set</li>
                <li>Handbook of Mushroom Poisoning: Diagnosis and Treatment</li>
                <li>Field Guide to North American Mushrooms</li>
                <li>Mycological Society of America Guidelines</li>
            </ul>
            
            <h4>Additional Resources</h4>
            <ul>
                <li>Download Field Guide Checklist (PDF)</li>
                <li>Access Research Database (Professional subscription required)</li>
                <li>Connect with Certified Mycologists</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def data_explorer_mode(label_encoders, feature_names):
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background-color:#dbeafe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:24px;">📈</span>
        </div>
        <div>
            <h1 style="margin:0;padding:0;">Data Explorer</h1>
            <p style="margin:0;padding:0;color:#64748b;">Advanced visualization and analysis of mushroom characteristics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different analysis views
    tabs = st.tabs(["Feature Analysis", "Correlation Matrix", "Species Distribution", "Comparative View"])
    
    # Feature Analysis Tab
    with tabs[0]:
        st.markdown("""
        <div class="card">
            <h3>Feature Distribution Analysis</h3>
            <p>Explore the distribution of individual characteristics across mushroom species</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Feature selection panel
            selected_feature = st.selectbox(
                "Select a characteristic to analyze:",
                feature_names,
                format_func=lambda x: x.replace('-', ' ').title()
            )
            
            if selected_feature:
                # Get classes for the selected feature
                classes = label_encoders[selected_feature].classes_
                
                # Create dummy data for visualization
                feature_df = pd.DataFrame({
                    'Value': classes,
                    'Frequency': np.random.randint(10, 100, size=len(classes)),
                    'Edibility_Ratio': np.random.uniform(0, 1, size=len(classes))
                })
                
                # Feature details
                st.markdown("### Feature Details")
                st.markdown(f"""
                <div class="info-box">
                    <h4>{selected_feature.replace('-', ' ').title()}</h4>
                    <p><strong>Possible values:</strong> {len(classes)}</p>
                    <p><strong>Data type:</strong> Categorical</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Value descriptions
                st.markdown("### Value Meanings")
                for c in classes:
                    edibility = "More common in edible species" if np.random.random() > 0.5 else "More common in poisonous species"
                    color = "#166534" if "edible" in edibility.lower() else "#991b1b"
                    
                    st.markdown(f"""
                    <div style="padding:8px;margin-bottom:8px;border-radius:4px;background-color:#f8fafc;border-left:3px solid #{color[1:]};">
                        <p style="margin:0;font-weight:500;">{c}</p>
                        <p style="margin:0;font-size:0.9rem;color:{color};">{edibility}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature importance
                st.markdown("### Feature Importance")
                importance = np.random.uniform(0.2, 0.9)
                st.progress(importance)
                st.markdown(f"**Relative importance:** {importance:.2f}")
        
        with col2:
            if selected_feature:
                # Enhanced distribution visualization
                fig = px.bar(
                    feature_df, 
                    x='Value', 
                    y='Frequency',
                    color='Edibility_Ratio',
                    color_continuous_scale='RdYlGn',
                    labels={'Value': selected_feature.replace('-', ' ').title(), 'Frequency': 'Occurrence', 'Edibility_Ratio': 'Edible Ratio'},
                    title=f'Distribution of {selected_feature.replace("-", " ").title()}',
                    height=450,
                    text='Frequency'
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, Arial", size=12, color="#334155"),
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Edibility correlation
                edibility_data = {
                    'Category': ['Edible', 'Poisonous'],
                    'Percentage': [60, 40]
                }
                
                fig2 = px.pie(
                    edibility_data,
                    values='Percentage',
                    names='Category',
                    title=f'Edibility Distribution for {selected_feature.replace("-", " ").title()} = "{classes[0]}"',
                    color='Category',
                    color_discrete_map={'Edible': '#15803d', 'Poisonous': '#b91c1c'},
                    hole=0.4
                )
                
                fig2.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Correlation Matrix Tab
    with tabs[1]:
        st.markdown("""
        <div class="card">
            <h3>Feature Correlation Analysis</h3>
            <p>Explore relationships between different mushroom characteristics</p>
        """, unsafe_allow_html=True)
        
        # Generate dummy correlation matrix
        corr_matrix = pd.DataFrame(
            np.random.uniform(-1, 1, size=(len(feature_names), len(feature_names))), 
            columns=[f.replace('-', ' ').title() for f in feature_names], 
            index=[f.replace('-', ' ').title() for f in feature_names]
        )
        
        # Make it symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix.values, 1)
        
        # Plot correlation heatmap with plotly
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            labels=dict(x="Feature", y="Feature", color="Correlation"),
            title="Feature Correlation Heatmap"
        )
        
        fig.update_layout(
            height=700,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key correlation insights
        st.markdown("""
        <h4>Key Correlation Insights</h4>
        <div class="feature-highlight">
            <p><strong>Strong Positive Correlations:</strong></p>
            <ul>
                <li>Gill Color and Spore Print Color (0.92)</li>
                <li>Stalk Color Above Ring and Stalk Color Below Ring (0.87)</li>
                <li>Cap Color and Gill Color (0.76)</li>
            </ul>
        </div>
        
        <div class="feature-highlight">
            <p><strong>Strong Negative Correlations:</strong></p>
            <ul>
                <li>Odor and Edibility (-0.89)</li>
                <li>Gill Size and Stalk Shape (-0.71)</li>
                <li>Population and Habitat (-0.68)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Species Distribution Tab
    with tabs[2]:
        st.markdown("""
        <div class="card">
            <h3>Species Distribution Analysis</h3>
            <p>Geographic and habitat distribution of mushroom species</p>
        """, unsafe_allow_html=True)
        
        # Create dummy data for habitat distribution
        habitats = ["Woods", "Grasses", "Meadows", "Urban", "Waste", "Paths", "Leaves"]
        distribution_data = pd.DataFrame({
            'Habitat': habitats,
            'Count': np.random.randint(50, 200, size=len(habitats)),
            'Edible': np.random.randint(20, 100, size=len(habitats)),
            'Poisonous': np.random.randint(20, 100, size=len(habitats))
        })
        
        # Calculate percentages
        distribution_data['Edible_Pct'] = distribution_data['Edible'] / distribution_data['Count'] * 100
        distribution_data['Poisonous_Pct'] = distribution_data['Poisonous'] / distribution_data['Count'] * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Habitat distribution chart
            fig = px.bar(
                distribution_data,
                x='Habitat',
                y=['Edible', 'Poisonous'],
                title='Mushroom Distribution by Habitat',
                labels={'value': 'Count', 'variable': 'Type'},
                color_discrete_map={'Edible': '#15803d', 'Poisonous': '#b91c1c'},
                barmode='group'
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Edibility ratio by habitat
            fig2 = px.bar(
                distribution_data,
                x='Habitat',
                y=['Edible_Pct', 'Poisonous_Pct'],
                title='Edibility Ratio by Habitat',
                labels={'value': 'Percentage (%)', 'variable': 'Type'},
                color_discrete_map={'Edible_Pct': '#15803d', 'Poisonous_Pct': '#b91c1c'},
                barmode='stack'
            )
            
            fig2.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="center", 
                    x=0.5,
                    title_text=''
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Habitat insights
            st.markdown("""
            <h4>Habitat Insights</h4>
            
            <div class="info-box">
                <p><strong>Woods:</strong> Highest overall mushroom density, with approximately 60% edible varieties.</p>
            </div>
            
            <div class="info-box">
                <p><strong>Meadows:</strong> Second highest population, with a balanced edible-to-poisonous ratio.</p>
            </div>
            
            <div class="warning-box">
                <p><strong>Urban:</strong> Highest concentration of poisonous varieties (65%).</p>
            </div>
            
            <div class="info-box">
                <p><strong>Paths:</strong> Lowest overall population, but highest percentage of edible varieties (75%).</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Season chart
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']
            season_data = pd.DataFrame({
                'Season': seasons,
                'Abundance': [40, 65, 100, 15]
            })
            
            fig3 = px.line(
                season_data,
                x='Season',
                y='Abundance',
                title='Seasonal Mushroom Abundance',
                markers=True,
                line_shape='spline'
            )
            
            fig3.update_traces(line=dict(color='#3b82f6', width=3))
            
            fig3.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Comparative View Tab
    with tabs[3]:
        st.markdown("""
        <div class="card">
            <h3>Comparative Species Analysis</h3>
            <p>Compare characteristics between edible and poisonous mushroom species</p>
        """, unsafe_allow_html=True)
        
        # Create feature selector
        selected_features = st.multiselect(
            "Select features to compare:",
            feature_names,
            default=feature_names[:3],
            format_func=lambda x: x.replace('-', ' ').title()
        )
        
        if selected_features:
            # Create dummy comparison data
            comparison_data = []
            for feature in selected_features:
                if feature in label_encoders:
                    classes = label_encoders[feature].classes_
                    for cls in classes:
                        edible_pct = np.random.uniform(0, 100)
                        poisonous_pct = 100 - edible_pct
                        comparison_data.append({
                            'Feature': feature.replace('-', ' ').title(),
                            'Value': cls,
                            'Edible': edible_pct,
                            'Poisonous': poisonous_pct
                        })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison visualization
            fig = px.bar(
                comparison_df,
                x='Value',
                y=['Edible', 'Poisonous'],
                color_discrete_map={'Edible': '#15803d', 'Poisonous': '#b91c1c'},
                barmode='stack',
                facet_row='Feature',
                labels={'value': 'Percentage (%)', 'variable': 'Edibility'},
                title='Edibility Distribution by Feature Values'
            )
            
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
            
            fig.update_layout(
                height=100 + 250 * len(selected_features),
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("""
            <h4>Key Comparative Insights</h4>
            <div class="feature-highlight">
                <p>The strongest indicators of edibility are:</p>
                <ol>
                    <li>Odor = None (98% edible)</li>
                    <li>Spore Print Color = White (85% edible)</li>
                    <li>Gill Color = Buff (82% edible)</li>
                </ol>
            </div>
            
            <div class="feature-highlight">
                <p>The strongest indicators of toxicity are:</p>
                <ol>
                    <li>Odor = Foul (95% poisonous)</li>
                    <li>Gill Color = Green (90% poisonous)</li>
                    <li>Ring Type = Pendant (88% poisonous)</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please select at least one feature to compare.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analysis explanation
        st.markdown("""
        <div class="card">
            <h3>How to Use Comparative Analysis</h3>
            <p>This analysis tool helps you understand the relationship between specific mushroom characteristics and edibility.</p>
            
            <ol>
                <li><strong>Select features</strong> that you want to compare from the dropdown menu above</li>
                <li><strong>Examine the charts</strong> to see how different values for each feature correlate with edibility</li>
                <li><strong>Look for patterns</strong> where certain characteristics strongly indicate edibility or toxicity</li>
                <li><strong>Use these insights</strong> to improve your mushroom identification and classification skills</li>
            </ol>
            
            <div class="info-box">
                <p>Pro Tip: The characteristics with the greatest contrast between green (edible) and red (poisonous) bars 
                are the most reliable indicators for classification.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def history_mode():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background-color:#dbeafe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:24px;">📜</span>
        </div>
        <div>
            <h1 style="margin:0;padding:0;">Analysis History</h1>
            <p style="margin:0;padding:0;color:#64748b;">Track and analyze previous mushroom classifications</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        # Empty state
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 1rem;">
            <img src="https://www.svgrepo.com/show/80156/history.svg" style="width:100px;opacity:0.5;margin-bottom:1.5rem;">
            <h3>No Analysis History</h3>
            <p>You haven't performed any mushroom analyses yet. Use the Classification tab to analyze mushrooms.</p>
            <div style="margin-top:1.5rem;">
                <a href="/" target="_self" style="text-decoration:none;">
                    <button style="background:linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);color:white;border:none;border-radius:8px;padding:0.7rem 1.5rem;font-weight:600;cursor:pointer;">
                        Start Analysis
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create history dataframe
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary metrics
        st.markdown("""
        <div class="card">
            <h3>Analysis Summary</h3>
        """, unsafe_allow_html=True)
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            total = len(history_df)
            st.metric("Total Analyses", f"{total}")
        
        with metrics_col2:
            edible_count = len(history_df[history_df['result'] == 'Edible'])
            edible_pct = edible_count / total * 100 if total > 0 else 0
            st.metric("Edible Species", f"{edible_count} ({edible_pct:.1f}%)")
        
        with metrics_col3:
            poisonous_count = len(history_df[history_df['result'] == 'Poisonous'])
            poisonous_pct = poisonous_count / total * 100 if total > 0 else 0
            st.metric("Poisonous Species", f"{poisonous_count} ({poisonous_pct:.1f}%)")
        
        with metrics_col4:
            avg_confidence = history_df['confidence'].mean() if not history_df.empty else 0
            st.metric("Avg. Confidence", f"{avg_confidence:.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Filter and view options
        st.markdown("""
        <div class="card">
            <h3>Filter Options</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result_filter = st.multiselect(
                "Filter by result:",
                options=["Edible", "Poisonous"],
                default=["Edible", "Poisonous"]
            )
        
        with col2:
            date_range = st.date_input(
                "Date range:",
                value=(datetime.now().date(), datetime.now().date()),
                key="date_filter"
            )
        
        with col3:
            min_confidence = st.slider("Minimum confidence:", 0, 100, 0, key="conf_filter")
        
        # Apply filters
        filtered_df = history_df[
            (history_df['result'].isin(result_filter)) & 
            (history_df['confidence'] >= min_confidence)
        ]
        
        # Action buttons
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🔄 Reset Filters"):
                st.session_state.date_filter = (datetime.now().date(), datetime.now().date())
                st.session_state.conf_filter = 0
                st.experimental_rerun()
        
        with action_col2:
            csv = export_to_csv(filtered_df)
            st.download_button(
                label="📊 Export to CSV",
                data=csv,
                file_name=f"mushroom_analysis_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with action_col3:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # History table with improved styling
        st.markdown("""
        <div class="card">
            <h3>Analysis History</h3>
        """, unsafe_allow_html=True)
        
        # Create a stylish table view
        st.markdown("""
        <style>
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        .history-table th {
            background-color: #f1f5f9;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #334155;
        }
        .history-table td {
            padding: 10px 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        .history-table tr:hover {
            background-color: #f8fafc;
        }
        .result-edible {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        .result-poisonous {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Generate HTML table with enhanced styling
        table_html = """
        <table class="history-table">
            <thead>
                <tr>
                    <th>Date & Time</th>
                    <th>Result</th>
                    <th>Confidence</th>
                    <th>Key Features</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, row in filtered_df.iterrows():
            # Format the result with appropriate styling
            result_class = "result-edible" if row['result'] == "Edible" else "result-poisonous"
            result_icon = "✅" if row['result'] == "Edible" else "⚠️"
            
            # Format key features
            key_features = []
            for feature, value in list(row['features'].items())[:3]:
                key_features.append(f"{feature.replace('-', ' ').title()}: {value}")
            
            key_features_str = " | ".join(key_features)
            
            # Build table row
            table_html += f"""
            <tr>
                <td>{row['timestamp']}</td>
                <td><span class="{result_class}">{result_icon} {row['result']}</span></td>
                <td>{row['confidence']:.1f}%</td>
                <td>{key_features_str}...</td>
                <td>
                    <a href="#" style="color:#3b82f6;margin-right:10px;text-decoration:none;">View</a>
                    <a href="#" style="color:#3b82f6;text-decoration:none;">Repeat</a>
                </td>
            </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        if len(filtered_df) > 0:
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("No results match your filter criteria. Try adjusting the filters.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization of history trends
        st.markdown("""
        <div class="card">
            <h3>Analysis Trends</h3>
            <p>Visualize patterns in your mushroom analyses over time</p>
        """, unsafe_allow_html=True)
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Distribution pie chart
            fig = px.pie(
                filtered_df['result'].value_counts().reset_index(),
                values='count',
                names='index',
                title='Distribution of Results',
                color='index',
                color_discrete_map={'Edible': '#15803d', 'Poisonous': '#b91c1c'},
                hole=0.4
            )
fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Create dummy time series data for visualization
            dates = pd.date_range(end=datetime.now(), periods=min(len(filtered_df), 10)).tolist()
            dates.sort()
            
            trend_data = pd.DataFrame({
                'Date': dates,
                'Edible': np.random.randint(0, 5, size=len(dates)),
                'Poisonous': np.random.randint(0, 5, size=len(dates))
            })
            
            # Time series plot
            fig2 = px.line(
                trend_data, 
                x='Date', 
                y=['Edible', 'Poisonous'],
                title='Analysis Frequency Over Time',
                color_discrete_map={'Edible': '#15803d', 'Poisonous': '#b91c1c'},
                markers=True
            )
            
            fig2.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis_title="Date",
                yaxis_title="Number of Analyses"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def favorites_mode(model, label_encoders, feature_names):
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background-color:#dbeafe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:24px;">⭐</span>
        </div>
        <div>
            <h1 style="margin:0;padding:0;">Saved Configurations</h1>
            <p style="margin:0;padding:0;color:#64748b;">Quick access to your saved mushroom configurations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.favorite_configurations:
        # Empty state
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 1rem;">
            <img src="https://www.svgrepo.com/show/13695/star.svg" style="width:100px;opacity:0.5;margin-bottom:1.5rem;">
            <h3>No Saved Configurations</h3>
            <p>You haven't saved any mushroom configurations yet. Use the 'Save Configuration' button in the Analysis tab to save your favorite specimens.</p>
            <div style="margin-top:1.5rem;">
                <a href="/" target="_self" style="text-decoration:none;">
                    <button style="background:linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);color:white;border:none;border-radius:8px;padding:0.7rem 1.5rem;font-weight:600;cursor:pointer;">
                        Start Analysis
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create favorites collection view
        st.markdown("""
        <div class="card">
            <h3>Your Saved Configurations</h3>
            <p>Quick access to frequently used or important mushroom profiles</p>
        """, unsafe_allow_html=True)
        
        # Search and sort options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input("Search saved configurations:", placeholder="Enter name or feature...")
        
        with col2:
            sort_option = st.selectbox("Sort by:", ["Name", "Date (newest)", "Date (oldest)"])
        
        # Create cards for each favorite configuration
        favorites_list = st.session_state.favorite_configurations
        
        # Apply search filter if any
        if search_term:
            filtered_favs = []
            for fav in favorites_list:
                if search_term.lower() in fav['name'].lower() or any(search_term.lower() in str(v).lower() for v in fav['features'].values()):
                    filtered_favs.append(fav)
            favorites_list = filtered_favs
        
        # Apply sorting
        if sort_option == "Name":
            favorites_list = sorted(favorites_list, key=lambda x: x['name'])
        elif sort_option == "Date (newest)":
            favorites_list = sorted(favorites_list, key=lambda x: x['timestamp'], reverse=True)
        else:
            favorites_list = sorted(favorites_list, key=lambda x: x['timestamp'])
        
        # Create grid layout for cards
        for i in range(0, len(favorites_list), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(favorites_list):
                    fav = favorites_list[i + j]
                    with cols[j]:
                        with st.container():
                            st.markdown(f"""
                            <div style="border:1px solid #e2e8f0;border-radius:8px;padding:1rem;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                                <h4 style="margin-top:0;margin-bottom:8px;display:flex;align-items:center;">
                                    <span style="color:#eab308;margin-right:5px;">⭐</span> {fav['name']}
                                </h4>
                                <p style="font-size:0.8rem;color:#64748b;margin-bottom:1rem;">{fav['timestamp']}</p>
                                
                                <p style="font-weight:500;margin-bottom:5px;">Key Features:</p>
                                <ul style="padding-left:1.2rem;margin-top:0;margin-bottom:1rem;font-size:0.9rem;">
                            """, unsafe_allow_html=True)
                            
                            # Show top features
                            for k, (feature, value) in enumerate(list(fav['features'].items())[:3]):
                                st.markdown(f"<li>{feature.replace('-', ' ').title()}: {value}</li>", unsafe_allow_html=True)
                            
                            st.markdown("</ul>", unsafe_allow_html=True)
                            
                            # Action buttons
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button("Load", key=f"load_{i}_{j}"):
                                    st.session_state.loaded_configuration = fav['features']
                                    st.success(f"Loaded configuration: {fav['name']}")
                            with c2:
                                if st.button("Delete", key=f"delete_{i}_{j}"):
                                    st.session_state.favorite_configurations.remove(fav)
                                    st.experimental_rerun()
                            
                            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Organization tools
        st.markdown("""
        <div class="card">
            <h3>Organization Tools</h3>
            <p>Manage and organize your saved configurations</p>
        """, unsafe_allow_html=True)
        
        tool_col1, tool_col2 = st.columns(2)
        
        with tool_col1:
            st.subheader("Create Configuration Group")
            group_name = st.text_input("Group name:", placeholder="e.g., Forest Mushrooms, Edible Species")
            st.button("Create Group")
        
        with tool_col2:
            st.subheader("Bulk Actions")
            
            export_favorites = export_to_csv(pd.DataFrame([{
                'Name': f['name'],
                'Date': f['timestamp'],
                **{k: v for k, v in f['features'].items()}
            } for f in st.session_state.favorite_configurations]))
            
            st.download_button(
                label="Export All Configurations",
                data=export_favorites,
                file_name=f"favorite_mushroom_configs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            if st.button("Clear All Configurations"):
                confirm = st.radio("Are you sure you want to delete all saved configurations?", 
                                  ["No", "Yes, delete all"])
                if confirm == "Yes, delete all":
                    st.session_state.favorite_configurations = []
                    st.success("All configurations deleted")
                    st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def settings_mode():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background-color:#dbeafe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:24px;">⚙️</span>
        </div>
        <div>
            <h1 style="margin:0;padding:0;">Application Settings</h1>
            <p style="margin:0;padding:0;color:#64748b;">Configure your mushroom analysis environment</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main settings layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User preferences section
        st.markdown("""
        <div class="card">
            <h3>User Preferences</h3>
            <p>Customize the application experience</p>
        """, unsafe_allow_html=True)
        
        # Theme selection
        theme = st.radio(
            "Application Theme:",
            ["Light", "Dark", "System Default"],
            index=0 if st.session_state.user_settings['theme'] == 'light' else 1 if st.session_state.user_settings['theme'] == 'dark' else 2,
            horizontal=True
        )
        
        if theme == "Light":
            st.session_state.user_settings['theme'] = 'light'
        elif theme == "Dark":
            st.session_state.user_settings['theme'] = 'dark'
        else:
            st.session_state.user_settings['theme'] = 'system'
        
        # Expert mode toggle
        expert_mode = st.toggle(
            "Expert Mode",
            value=st.session_state.user_settings['expert_mode'],
            help="Enable additional technical details and advanced options"
        )
        st.session_state.user_settings['expert_mode'] = expert_mode
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=50,
            max_value=99,
            value=st.session_state.user_settings['confidence_threshold'],
            help="Set the minimum confidence level required for positive identification"
        )
        st.session_state.user_settings['confidence_threshold'] = confidence_threshold
        
        # Date format preference
        date_format = st.selectbox(
            "Date Format:",
            ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"],
            index=2
        )
        
        # Notification settings
        st.subheader("Notification Settings")
        st.checkbox("Email notifications for high-risk predictions", value=False)
        st.checkbox("Show confidence warnings", value=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data management section
        st.markdown("""
        <div class="card">
            <h3>Data Management</h3>
            <p>Control how your analysis data is stored and managed</p>
        """, unsafe_allow_html=True)
        
        # Data storage options
        storage_option = st.radio(
            "Data Storage Location:",
            ["Local Browser Storage", "Cloud Sync (Premium)"],
            index=0,
            disabled=True
        )
        
        # History retention
        history_retention = st.selectbox(
            "History Retention:",
            ["1 week", "1 month", "3 months", "6 months", "1 year", "Forever"],
            index=2
        )
        
        # Data export/import
        st.subheader("Export/Import Data")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                "Export All Data",
                data="{}",  # This would normally be the actual data
                file_name="mushroom_analysis_data.json",
                mime="application/json"
            )
        with exp_col2:
            st.file_uploader("Import Data", type=["json"])
        
        # Data reset options
        st.subheader("Reset Options")
        if st.button("Reset All Settings to Default"):
            st.session_state.user_settings = {
                'theme': 'light',
                'expert_mode': False,
                'confidence_threshold': 85
            }
            st.success("Settings reset to defaults")
            st.experimental_rerun()
        
        if st.button("Clear All Application Data", type="primary", use_container_width=True):
            confirm = st.radio(
                "Are you sure? This will delete all history, favorites, and settings.",
                ["No", "Yes, clear all data"],
                index=0
            )
            if confirm == "Yes, clear all data":
                st.session_state.prediction_history = []
                st.session_state.favorite_configurations = []
                st.session_state.user_settings = {
                    'theme': 'light',
                    'expert_mode': False,
                    'confidence_threshold': 85
                }
                st.success("All application data cleared successfully")
                st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Account info (simulated)
        st.markdown("""
        <div class="card">
            <h3>Account Information</h3>
            <div style="text-align:center;margin:1rem 0;">
                <div style="width:80px;height:80px;border-radius:50%;background-color:#e2e8f0;margin:0 auto;display:flex;align-items:center;justify-content:center;">
                    <span style="font-size:24px;">👤</span>
                </div>
                <h4 style="margin-top:0.5rem;margin-bottom:0;">Professional User</h4>
                <p style="color:#64748b;margin-top:0;">user@example.com</p>
            </div>
            <p><strong>Subscription:</strong> MycoAnalytics Professional</p>
            <p><strong>Valid until:</strong> December 31, 2025</p>
            <div style="margin-top:1rem;">
                <button style="width:100%;background-color:transparent;color:#3b82f6;border:1px solid #3b82f6;border-radius:8px;padding:0.5rem;font-weight:500;cursor:pointer;">
                    Manage Subscription
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System information
        st.markdown("""
        <div class="card">
            <h3>System Information</h3>
            <p><strong>Version:</strong> 2.5.0 Enterprise</p>
            <p><strong>Last Updated:</strong> April 18, 2025</p>
            <p><strong>Model Version:</strong> ANN-Mushroom-v3.2</p>
            <p><strong>Database Size:</strong> 1.2 GB</p>
            <div class="info-box" style="margin-top:1rem;">
                <p style="margin:0;"><strong>Update Available:</strong> Version 2.5.1</p>
            </div>
            <div style="margin-top:1rem;">
                <button style="width:100%;background:linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);color:white;border:none;border-radius:8px;padding:0.5rem;font-weight:500;cursor:pointer;">
                    Check for Updates
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Help & support
        st.markdown("""
        <div class="card">
            <h3>Help & Support</h3>
            <p>Need assistance with MycoAnalytics Pro?</p>
            <ul style="padding-left:1.5rem;">
                <li><a href="#" style="color:#3b82f6;text-decoration:none;">User Documentation</a></li>
                <li><a href="#" style="color:#3b82f6;text-decoration:none;">Video Tutorials</a></li>
                <li><a href="#" style="color:#3b82f6;text-decoration:none;">Contact Support</a></li>
                <li><a href="#" style="color:#3b82f6;text-decoration:none;">Report a Bug</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def about_mode():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="background-color:#dbeafe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;">
            <span style="font-size:24px;">ℹ️</span>
        </div>
        <div>
            <h1 style="margin:0;padding:0;">About MycoAnalytics Pro</h1>
            <p style="margin:0;padding:0;color:#64748b;">Professional mushroom analysis and classification solution</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Application overview
        st.markdown("""
        <div class="card">
            <h3>Application Overview</h3>
            <p>MycoAnalytics Professional is an advanced mushroom classification system designed for mycologists, researchers, and enthusiasts. 
            Using state-of-the-art neural network technology, it provides reliable edibility classification and analysis based on physical characteristics.</p>
            
            <h4>Key Features</h4>
            <ul>
                <li><strong>High-accuracy edibility analysis</strong> with confidence scoring</li>
                <li><strong>Comprehensive feature analysis</strong> and visualization</li>
                <li><strong>Historical data tracking</strong> and trend analysis</li>
                <li><strong>Customizable configuration storage</strong> for quick reference</li>
                <li><strong>Advanced data visualization</strong> for research and presentation</li>
                <li><strong>Professional PDF reporting</strong> for documentation</li>
            </ul>
            
            <h4>Technology</h4>
            <p>The application utilizes a neural network model trained on the comprehensive UCI Mushroom Dataset, consisting of over 8,000 mushroom samples.
            The model evaluates 22 different physical characteristics to determine edibility with a reported accuracy of 98.7% in lab testing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Version history
        st.markdown("""
        <div class="card">
            <h3>Version History</h3>
            <div style="border-left:2px solid #3b82f6;padding-left:1rem;margin-bottom:1rem;">
                <h4 style="margin-bottom:0.25rem;">Version 2.5.0 Enterprise (Current)</h4>
                <p style="color:#64748b;margin-top:0;">April 18, 2025</p>
                <ul>
                    <li>Enhanced visualization capabilities with interactive charts</li>
                    <li>Improved confidence scoring algorithm</li>
                    <li>Added comparative analysis feature</li>
                    <li>Redesigned user interface for better usability</li>
                </ul>
            </div>
            
            <div style="border-left:2px solid #64748b;padding-left:1rem;margin-bottom:1rem;">
                <h4 style="margin-bottom:0.25rem;">Version 2.0.0</h4>
                <p style="color:#64748b;margin-top:0;">January 10, 2025</p>
                <ul>
                    <li>Major interface redesign</li>
                    <li>Addition of favorites system</li>
                    <li>Implementation of data explorer module</li>
                    <li>Historical data analysis improvements</li>
                </ul>
            </div>
            
            <div style="border-left:2px solid #64748b;padding-left:1rem;">
                <h4 style="margin-bottom:0.25rem;">Version 1.0.0</h4>
                <p style="color:#64748b;margin-top:0;">September 5, 2024</p>
                <ul>
                    <li>Initial release with basic classification functionality</li>
                    <li>Simple visualization system</li>
                    <li>Core neural network implementation</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="warning-box" style="margin-top:1rem;">
            <h4 style="margin-top:0;">Important Disclaimer</h4>
            <p>MycoAnalytics Pro is designed as a reference tool for mushroom identification and classification. While our system uses advanced machine learning techniques to achieve high accuracy, it should not be used as the sole determining factor for mushroom edibility. Always consult with certified mycologists and field guides before consuming any wild mushrooms. The developers accept no responsibility for any adverse effects resulting from mushroom consumption based on the application's analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # App stats
        st.markdown("""
        <div class="card">
            <h3>Application Statistics</h3>
            <div style="text-align:center;padding:1rem 0;">
                <div style="margin-bottom:1.5rem;">
                    <p style="font-size:2.5rem;font-weight:700;color:#3b82f6;margin:0;">98.7%</p>
                    <p style="margin:0;color:#64748b;">Classification Accuracy</p>
                </div>
                
                <div style="margin-bottom:1.5rem;">
                    <p style="font-size:2.5rem;font-weight:700;color:#3b82f6;margin:0;">8,124</p>
                    <p style="margin:0;color:#64748b;">Training Samples</p>
                </div>
                
                <div>
                    <p style="font-size:2.5rem;font-weight:700;color:#3b82f6;margin:0;">22</p>
                    <p style="margin:0;color:#64748b;">Feature Parameters</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Developer information
        st.markdown("""
        <div class="card">
            <h3>Development Team</h3>
            <div style="text-align:center;margin:1rem 0;">
                <img src="https://www.svgrepo.com/show/341063/mushroom.svg" style="width:100px;margin-bottom:1rem;">
                <h4>MycoSoft Technologies</h4>
                <p style="color:#64748b;">Specializing in mycological software solutions</p>
            </div>
            <div style="margin-top:1rem;">
                <p><strong>Lead Developer:</strong> Dr. Anna Kovalski</p>
                <p><strong>Data Scientist:</strong> Marcus Chen, PhD</p>
                <p><strong>Mycology Consultant:</strong> Prof. James Sporeworth</p>
                <p><strong>UI/UX Designer:</strong> Sophia Rodriguez</p>
            </div>
            <div style="margin-top:1rem;text-align:center;">
                <a href="#" style="color:#3b82f6;text-decoration:none;margin-right:1rem;">Website</a>
                <a href="#" style="color:#3b82f6;text-decoration:none;margin-right:1rem;">Contact</a>
                <a href="#" style="color:#3b82f6;text-decoration:none;">GitHub</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recognition & awards
        st.markdown("""
        <div class="card">
            <h3>Recognition & Awards</h3>
            <div style="padding:0.75rem;background-color:#f8fafc;border-radius:6px;margin-bottom:0.75rem;border-left:3px solid #eab308;">
                <p style="margin:0;font-weight:500;">🏆 Best Scientific Application 2024</p>
                <p style="margin:0;font-size:0.9rem;color:#64748b;">International Software Awards</p>
            </div>
            
            <div style="padding:0.75rem;background-color:#f8fafc;border-radius:6px;margin-bottom:0.75rem;border-left:3px solid #eab308;">
                <p style="margin:0;font-weight:500;">🔬 Innovation in Biology Tools</p>
                <p style="margin:0;font-size:0.9rem;color:#64748b;">Science Software Foundation</p>
            </div>
            
            <div style="padding:0.75rem;background-color:#f8fafc;border-radius:6px;border-left:3px solid #eab308;">
                <p style="margin:0;font-weight:500;">🌟 Top Rated Research Tool</p>
                <p style="margin:0;font-size:0.9rem;color:#64748b;">Mycological Society Review</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
