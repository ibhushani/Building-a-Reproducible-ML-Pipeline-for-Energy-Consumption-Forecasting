import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page config
st.set_page_config(
    page_title="⚡ Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚡ Energy Consumption Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict household energy consumption using Machine Learning")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Power Measurements")
    
    global_reactive_power = st.slider(
        "Global Reactive Power (kW)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )
    
    voltage = st.slider(
        "Voltage (V)",
        min_value=220.0, max_value=255.0, value=235.0, step=0.1
    )
    
    global_intensity = st.slider(
        "Global Intensity (A)",
        min_value=0.0, max_value=50.0, value=10.0, step=0.1
    )
    
    st.markdown("### 🔌 Sub-Metering")
    
    sub_metering_1 = st.number_input(
        "Sub Metering 1 - Kitchen (Wh)",
        min_value=0.0, max_value=100.0, value=0.0, step=1.0
    )
    
    sub_metering_2 = st.number_input(
        "Sub Metering 2 - Laundry (Wh)",
        min_value=0.0, max_value=100.0, value=1.0, step=1.0
    )
    
    sub_metering_3 = st.number_input(
        "Sub Metering 3 - Water Heater/AC (Wh)",
        min_value=0.0, max_value=50.0, value=17.0, step=1.0
    )

with col2:
    st.markdown("### 🕐 Time Information")
    
    use_current_time = st.checkbox("Use Current Time", value=True)
    
    if use_current_time:
        now = datetime.now()
        hour = now.hour
        day = now.day
        month = now.month
        weekday = now.weekday()
        st.info(f"📅 Current: {now.strftime('%Y-%m-%d %H:%M')}")
    else:
        hour = st.slider("Hour (0-23)", min_value=0, max_value=23, value=12)
        day = st.slider("Day of Month (1-31)", min_value=1, max_value=31, value=15)
        month = st.slider("Month (1-12)", min_value=1, max_value=12, value=6)
        weekday = st.slider("Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
    
    st.markdown("### ⚡ Make Prediction")
    
    if st.button("🔮 Predict Energy Consumption", use_container_width=True):
        # Create input data
        input_data = pd.DataFrame({
            'Global_reactive_power': [global_reactive_power],
            'Voltage': [voltage],
            'Global_intensity': [global_intensity],
            'Sub_metering_1': [sub_metering_1],
            'Sub_metering_2': [sub_metering_2],
            'Sub_metering_3': [sub_metering_3],
            'hour': [hour],
            'day': [day],
            'month': [month],
            'weekday': [weekday]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Energy Consumption</h2>
            <h1 style="font-size: 4rem; margin: 1rem 0;">{prediction:.3f} kW</h1>
            <p>Global Active Power</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input summary
        st.markdown("### 📋 Input Summary")
        st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Built with ❤️ using Streamlit | ML Pipeline with DVC & MLflow</p>
</div>
""", unsafe_allow_html=True)
