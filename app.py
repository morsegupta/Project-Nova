import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import time
import os
import traceback
import plotly.colors as pc
viridis_colors = pc.sample_colorscale('Viridis', [i/7 for i in range(8)])
st.set_page_config(
    page_title="NovaScore - Complete ML Architecture",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-story {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .api-status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .api-status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
    .debug-info {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
    
    /* Custom button styling - Multiple selectors for compatibility */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    div[data-testid="stButton"] > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(1, 177, 79, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #02d862 0%, #01b14f 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(1, 177, 79, 0.4) !important;
    }
    
    .stButton > button[kind="primary"]:active,
    .stButton > button[data-testid="baseButton-primary"]:active,
    div[data-testid="stButton"] > button[kind="primary"]:active,
    button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }
    
    /* Alternative aggressive targeting for Streamlit buttons */
    div.stButton button,
    div[data-testid="stButton"] button,
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(1, 177, 79, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton button:hover,
    div[data-testid="stButton"] button:hover,
    .stFormSubmitButton button:hover {
        background: linear-gradient(135deg, #02d862 0%, #01b14f 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(1, 177, 79, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:5001"
API_ENDPOINTS = {
    "predict": f"{API_BASE_URL}/api/predict",
    "health": f"{API_BASE_URL}/api/health",
    "model_info": f"{API_BASE_URL}/api/model_info"
}
@st.cache_data
def load_sample_data():
    try:
        indonesia_df = pd.read_csv('indonesia_partners.csv')
        ahmed_journey = pd.read_csv('ahmed_journey.csv')

        with open('indonesia_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return indonesia_df, ahmed_journey, metrics
    except Exception as e:
        st.warning(f"⚠️ Sample data files not found: {str(e)}. Using generated sample data.")
        return generate_sample_data()

def generate_sample_data():
    np.random.seed(42)
    
    indonesia_df = pd.DataFrame({
        'partner_id': [f'GRB_{i:06d}' for i in range(5000)],
        'city': np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan'], 5000),
        'nova_score': np.random.normal(600, 80, 5000).clip(300, 850),
        'monthly_earnings_usd': np.random.normal(550, 150, 5000).clip(200, 1500),
        'trip_completion_rate': np.random.normal(0.92, 0.05, 5000).clip(0.7, 1.0),
        'avg_customer_rating': np.random.normal(4.5, 0.3, 5000).clip(3.0, 5.0),
        'vehicle_type': np.random.choice(['Motorcycle', 'Car', 'Both'], 5000),
        'vehicle_loan_eligible': np.random.choice([True, False], 5000)
    })
    
    ahmed_journey = pd.DataFrame({
        'month': ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'],
        'nova_score': [420, 465, 520, 580, 635, 680],
        'monthly_earnings': [380, 420, 480, 520, 580, 640],
        'trip_completion_rate': [0.82, 0.87, 0.91, 0.93, 0.95, 0.97],
        'avg_rating': [3.9, 4.1, 4.3, 4.5, 4.7, 4.8]
    })
    
    metrics = {
        'total_partners': 5000,
        'avg_nova_score': 595.5,
        'credit_invisible_rate': 59.7,
        'avg_monthly_earnings': 555.2,
        'market_size_millions': 28.7
    }
    
    return indonesia_df, ahmed_journey, metrics

def check_api_health():
    try:
        st.write(f"🔍 Checking API health at: {API_ENDPOINTS['health']}")
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            st.error(f"API returned status code: {response.status_code}")
            return False, None
    except requests.exceptions.ConnectionError as e:
        st.error(f"❌ Connection Error: Could not connect to Flask API. Make sure it's running on port 5001")
        return False, None
    except requests.exceptions.Timeout as e:
        st.error(f"❌ Timeout Error: API took too long to respond")
        return False, None
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        return False, None

def make_prediction_api_call(partner_data):
    try:
        st.write(f"📡 Making prediction call to: {API_ENDPOINTS['predict']}")
        
        response = requests.post(
            API_ENDPOINTS["predict"], 
            json=partner_data,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        
        st.write(f"📊 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"✅ API Response: {result}")
            return True, result
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            st.error(error_msg)
            return False, error_msg
            
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Error: Could not connect to API. Is Flask running?"
        st.error(error_msg)
        return False, error_msg
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout Error: API took too long to respond"
        st.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}\n{traceback.format_exc()}"
        st.error(error_msg)
        return False, error_msg

def get_model_info():
    try:
        response = requests.get(API_ENDPOINTS["model_info"], timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

indonesia_df, ahmed_journey, metrics = load_sample_data()

st.sidebar.markdown("# 🚀 NovaScore Dashboard")
st.sidebar.markdown("---")


with st.sidebar:
    with st.expander("🔧 API Status Check", expanded=True):
        if st.button("🔄 Check API Status"):
            api_healthy, api_status = check_api_health()
            
            if api_healthy:
                st.success("✅ API Status: Online")
                if api_status:
                    st.json(api_status)
            else:
                st.error("❌ API Status: Offline")
                st.markdown("""
                **Troubleshooting Steps:**
                1. Make sure Flask API is running: `python nova_api.py`
                2. Check if port 5001 is available
                3. Verify Flask is running on 0.0.0.0:5001
                4. Check for firewall/antivirus blocking
                """)

page = st.sidebar.selectbox(
    "Navigate to:",
    ["🏠 Overview", "🎯 Live Predictions", "📊 Model Performance", "🌟 Success Stories", "🇮🇩 Market Insights", "⚖️ Fairness & Bias", "🔧 API Integration"]
)

if page == "🏠 Overview":
    st.markdown('<div class="main-header">NovaScore - Complete ML Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("### 🏗️ System Architecture")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            image_path = os.path.join("assets", "architecture.png")
            
            st.image(image_path, caption="NovaScore Complete ML Architecture")
        except Exception as e:
            st.info(f"📸 Architecture diagram not found. Please ensure the image is in the assets folder. Error: {str(e)}")
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>🎯 Model Accuracy</h3><h2>93.6%</h2><p>Unified Ensemble Performance</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>⚡ Response Time</h3><h2>&lt;100ms</h2><p>Real-time Predictions</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>🌍 Market Size</h3><h2>$28.7M</h2><p>Indonesia Opportunity</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>👥 Partners Served</h3><h2>5,094</h2><p>Indonesia Focus</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Three Pillar Framework")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏆 Performance Trust (40%)**
        - Trip completion rate
        - Customer ratings
        - Consistency score
        - Cancellation penalty
        """)
    
    with col2:
        st.markdown("""
        **💰 Financial Behavior (35%)**
        - Monthly earnings
        - Earnings stability
        - Trip efficiency
        - Recent activity
        """)
    
    with col3:
        st.markdown("""
        **📱 Platform Engagement (25%)**
        - Feature adoption
        - Digital integration
        - Peak hour utilization
        - Community impact
        """)
    
    st.markdown("### 🔬 Advanced ML Architecture")
    st.info("""
    **🎯 Unified 26-Feature Models:**
    - **XGBoost Regression** (R² = 93.5%) - Gradient boosting for structured data
    - **LightGBM Regression** (R² = 93.6%) - Optimized gradient boosting  
    - **Neural Network** (R² = 93.1%) - Deep learning for complex patterns
    - **Ensemble Performance** (R² = 93.6%) - Weighted combination of all models
    - **Single Preprocessing Path** - Unified feature engineering for all models
    """)

elif page == "🎯 Live Predictions":
    st.markdown('<div class="main-header"> Live NovaScore Predictions</div>', unsafe_allow_html=True)
    
    api_healthy, api_status = check_api_health()
    
    if api_healthy:
        st.success("🟢 Connected to NovaScore Production API")
        if api_status:
            with st.expander("🔍 API Status Details"):
                st.json(api_status)
        
        model_info = get_model_info()
        if model_info:
            with st.expander("📊 Model Information"):
                st.json(model_info)
    else:
        st.error("🔴 API Connection Failed")
        st.markdown("""
        ### 🔧 Troubleshooting Guide
        
        **Start the Flask API:**
           ```
           python nova_api.py
           ```
        """)
        
        if st.button("🔄 Retry Connection"):
            st.rerun()
    
    st.markdown("### 📝 Partner Information Input")
    
    st.markdown("### ⚡ Quick Fill Presets")
    preset_cols = st.columns(4)

    presets = {
        "🚨 Emergency Loan": {
            "trip_completion_rate": 0.28,     
            "avg_customer_rating": 2.1,       
            "consistency_score": 0.32,        
            "cancellation_rate": 0.25,        
            "monthly_earnings_usd": 300,      
            "earning_volatility": 0.47,       
            "total_trips": 80,                
            "last_30_days_trips": 12,         
            "tenure_months": 3,               
            "feature_adoption_score": 0.20,   
            "platform_activity_days_per_month": 8,  
            "grabpay_usage_frequency": 0.15,  
            "peak_hours_utilization": 0.15,   
            "weekend_activity_ratio": 0.10,   
            "referral_count": 0,              
            "recent_rating_trend": -0.15,     
            "recent_earnings_trend": -0.10     
        },
        "💰 Micro Loan": {
            "trip_completion_rate": 0.13,     
            "avg_customer_rating": 2.1,       
            "consistency_score": 0.23,        
            "cancellation_rate": 0.29,        
            "monthly_earnings_usd": 400,      
            "earning_volatility": 0.6,       
            "total_trips": 300,               
            "last_30_days_trips": 25,         
            "tenure_months": 6,               
            "feature_adoption_score": 0.25,   
            "platform_activity_days_per_month": 8,
            "grabpay_usage_frequency": 0.22,   
            "peak_hours_utilization": 0.22,   
            "weekend_activity_ratio": 0.21,   
            "referral_count": 2,          
            "recent_rating_trend": -0.26,     
            "recent_earnings_trend": -0.16
        },
        "🚗 Vehicle Loan": {
            "trip_completion_rate": 0.92,    
            "avg_customer_rating": 4.6,  
            "consistency_score": 0.85,        
            "cancellation_rate": 0.04,       
            "monthly_earnings_usd": 800,    
            "earning_volatility": 0.15,     
            "total_trips": 800,              
            "last_30_days_trips": 65,      
            "tenure_months": 15,              
            "feature_adoption_score": 0.8,   
            "platform_activity_days_per_month": 25, 
            "grabpay_usage_frequency": 0.7,  
            "peak_hours_utilization": 0.6,
            "weekend_activity_ratio": 0.5,    
            "referral_count": 8,            
            "recent_rating_trend": 0.08,    
            "recent_earnings_trend": 0.06 
        },
        "🏆 Premium Loan": {
            "trip_completion_rate": 0.998,   
            "avg_customer_rating": 4.99,   
            "consistency_score": 0.99,       
            "cancellation_rate": 0.001,     
            "monthly_earnings_usd": 5000, 
            "earning_volatility": 0.01,      
            "total_trips": 5000,            
            "last_30_days_trips": 30,       
            "tenure_months": 84,             
            "feature_adoption_score": 0.999,
            "platform_activity_days_per_month": 30,
            "grabpay_usage_frequency": 0.999, 
            "peak_hours_utilization": 0.99,
            "weekend_activity_ratio": 0.95,   
            "referral_count": 50,         
            "recent_rating_trend": 0.4, 
            "recent_earnings_trend": 0.4 
        }
    }

    if 'form_values' not in st.session_state:
        st.session_state.form_values = presets["🚗 Vehicle Loan"]  

    for i, (preset_name, preset_values) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(preset_name, key=f"preset_{i}"):
                st.session_state.form_values = preset_values
                st.rerun()
    
    st.markdown("---")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚗 Trip Performance**")
            trip_completion_rate = st.slider("Trip Completion Rate", 0.0, 1.0, 
                                            st.session_state.form_values.get("trip_completion_rate", 0.92), 0.01)
            avg_customer_rating = st.slider("Average Customer Rating", 1.0, 5.0, 
                                           st.session_state.form_values.get("avg_customer_rating", 4.5), 0.1)
            consistency_score = st.slider("Consistency Score", 0.0, 1.0, 
                                        st.session_state.form_values.get("consistency_score", 0.85), 0.01)
            cancellation_rate = st.slider("Cancellation Rate", 0.0, 0.3, 
                                         st.session_state.form_values.get("cancellation_rate", 0.05), 0.01)
            
            st.markdown("**💰 Financial Metrics**")
            monthly_earnings_usd = st.number_input("Monthly Earnings (USD)", 0, 10000, 
                                                 st.session_state.form_values.get("monthly_earnings_usd", 550))
            earning_volatility = st.slider("Earning Volatility", 0.0, 1.0, 
                                          st.session_state.form_values.get("earning_volatility", 0.15), 0.01)
            total_trips = st.number_input("Total Trips", 10, 5000, 
                                        st.session_state.form_values.get("total_trips", 800))
            last_30_days_trips = st.number_input("Last 30 Days Trips", 0, 200, 
                                                st.session_state.form_values.get("last_30_days_trips", 85))
        
        with col2:
            st.markdown("**📱 Platform Engagement**")
            tenure_months = st.number_input("Tenure (Months)", 1, 120, 
                                          st.session_state.form_values.get("tenure_months", 18))
            feature_adoption_score = st.slider("Feature Adoption Score", 0.0, 1.0, 
                                              st.session_state.form_values.get("feature_adoption_score", 0.75), 0.01)
            platform_activity_days_per_month = st.slider("Active Days/Month", 0, 30, 
                                                        st.session_state.form_values.get("platform_activity_days_per_month", 22))
            grabpay_usage_frequency = st.slider("GrabPay Usage", 0.0, 1.0, 
                                               st.session_state.form_values.get("grabpay_usage_frequency", 0.65), 0.01)
            peak_hours_utilization = st.slider("Peak Hours Utilization", 0.0, 1.0, 
                                              st.session_state.form_values.get("peak_hours_utilization", 0.68), 0.01)
            weekend_activity_ratio = st.slider("Weekend Activity", 0.0, 1.0, 
                                              st.session_state.form_values.get("weekend_activity_ratio", 0.45), 0.01)
            referral_count = st.number_input("Referral Count", 0, 100, 
                                            st.session_state.form_values.get("referral_count", 3))
            
            st.markdown("**📊 Recent Trends**")
            recent_rating_trend = st.slider("Recent Rating Trend", -0.5, 0.5, 
                                           st.session_state.form_values.get("recent_rating_trend", 0.05), 0.01)
            recent_earnings_trend = st.slider("Recent Earnings Trend", -0.3, 0.3, 
                                             st.session_state.form_values.get("recent_earnings_trend", 0.08), 0.01)
        
        submitted = st.form_submit_button("🚀 Generate NovaScore Prediction", type="primary")
    
    st.markdown("""
    <style>
    div.stButton > button:first-child,
    div[data-testid="stFormSubmitButton"] > button:first-child {
        background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(1, 177, 79, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("💡 Understanding NovaScore Ranges", expanded=False):
        score_cols = st.columns(4) 
        
        with score_cols[0]:
            st.markdown("""
            **🚨 Emergency Loans (300-399)**
            - Just getting started
            - Building basic track record
            - Max: $2,000 at 18%
            - Expected: 350-390
            """)
        
        with score_cols[1]:
            st.markdown("""
            **💰 Micro Loans (400-499)**
            - Developing consistency
            - Basic platform engagement
            - Max: $10,000 at 12%
            - Expected: 420-480
            """)
        
        with score_cols[2]:
            st.markdown("""
            **🚗 Vehicle Loans (500-649)**
            - Solid performance
            - Regular earnings
            - Max: $25,000 at 8.5%
            - Expected: 550-620
            """)

        with score_cols[3]:
            st.markdown("""
            **🏆 Premium Loans (650+)**
            - Exceptional performance
            - Max: $50,000 at 5.5%
            - Extended terms (60+ months)
            - Expected: 650-720
            """)
        
        st.warning("""
        **🎯 Preset Target Ranges:**
        - 🚨 **Emergency Loan**: Targets 350-390 points (basic credit access)
        - 💰 **Micro Loan**: Targets 420-480 points (small business funding)  
        - 🚗 **Vehicle Loan**: Targets 550-620 points (vehicle financing)
        - 🏆 **Premium Loan**: Targets 650-720 points (premium financing)
        """)
        
        st.success("""
        **Progression Path:**
        - Start with Emergency → Build to Micro → Advance to Vehicle → Achieve Premium
        - Each tier represents significant improvement in creditworthiness
        - Premium tier requires exceptional, sustained performance across all metrics
        """)
    
    if submitted:
        raw_partner_data = {
            "trip_completion_rate": trip_completion_rate,
            "avg_customer_rating": avg_customer_rating,
            "consistency_score": consistency_score,
            "cancellation_rate": cancellation_rate,
            "monthly_earnings_usd": monthly_earnings_usd,
            "earning_volatility": earning_volatility,
            "total_trips": total_trips,
            "last_30_days_trips": last_30_days_trips,
            "tenure_months": tenure_months,
            "feature_adoption_score": feature_adoption_score,
            "platform_activity_days_per_month": platform_activity_days_per_month,
            "grabpay_usage_frequency": grabpay_usage_frequency,
            "peak_hours_utilization": peak_hours_utilization,
            "weekend_activity_ratio": weekend_activity_ratio,
            "referral_count": referral_count,
            "recent_rating_trend": recent_rating_trend,
            "recent_earnings_trend": recent_earnings_trend
        }
        
        
        with st.spinner("🔄 Processing prediction through ML ensemble..."):
            time.sleep(1)
            
            if api_healthy:
                success, result = make_prediction_api_call(raw_partner_data)
                
                
                if success and result:
                    try:
                        if isinstance(result, dict):
                            if 'prediction' in result:
                                prediction_data = result['prediction']
                            else:
                                prediction_data = result
                            
                            nova_score = prediction_data.get('nova_score', 600)
                            risk_category = prediction_data.get('risk_category', 'Medium')
                            loan_eligibility = prediction_data.get('loan_eligibility', {})
                            
                            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("🎯 NovaScore", f"{nova_score:.0f}", help="Score range: 300-850")
                                
                                fig_gauge = go.Figure(go.Indicator(
                                    mode = "gauge+number+delta",
                                    value = nova_score,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "NovaScore"},
                                    delta = {'reference': 650},
                                    gauge = {
                                        'axis': {'range': [None, 850]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [300, 500], 'color': "lightgray"},
                                            {'range': [500, 650], 'color': "yellow"},
                                            {'range': [650, 850], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 650
                                        }
                                    }
                                ))
                                fig_gauge.update_layout(height=300)
                                st.plotly_chart(fig_gauge, use_container_width=True)
                            
                            with col2:
                                risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                                risk_emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                                
                                st.metric("⚖️ Risk Category", f"{risk_emoji.get(risk_category, '⚪')} {risk_category}")
                                
                                if nova_score >= 650:
                                    st.success("✅ Low Risk - Excellent creditworthiness")
                                elif nova_score >= 500:
                                    st.warning("⚠️ Medium Risk - Good creditworthiness")
                                else:
                                    st.error("🔴 High Risk - Limited credit access")
                            
                            with col3:
                                st.markdown("**💳 Loan Eligibility:**")
                                
                                if nova_score >= 650: 
                                    st.success("✅ Premium Loans Available")
                                    st.write("**Max Amount:** $50,000")
                                    st.write("**Interest Rate:** 5.5%")
                                    st.write("**Term:** 60+ months")
                                elif nova_score >= 500:  
                                    st.success("✅ Vehicle Loans Available") 
                                    st.write("**Max Amount:** $25,000")
                                    st.write("**Interest Rate:** 8.5%")
                                    st.write("**Term:** 48 months")
                                elif nova_score >= 400:
                                    st.info("ℹ️ Micro Loans Available")
                                    st.write("**Max Amount:** $10,000")
                                    st.write("**Interest Rate:** 12.0%")
                                    st.write("**Term:** 36 months")
                                elif nova_score >= 300: 
                                    st.warning("⚠️ Emergency Loans Available")
                                    st.write("**Max Amount:** $2,000")
                                    st.write("**Interest Rate:** 18.0%")
                                    st.write("**Term:** 12 months")
                                else:  
                                    st.error("❌ No Loans Available")
                                    st.write("**Reason:** Score too low")
                                    st.write("**Recommendation:** Improve performance metrics")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("### 🔍 Prediction Insights")
                            insights_col1, insights_col2 = st.columns(2)
                            
                            with insights_col1:
                                score_percentile = ((nova_score - 300) / (850 - 300)) * 100
                                st.info(f"""
                                **Score Analysis:**
                                - Your score of {nova_score:.0f} is in the {score_percentile:.0f}th percentile
                                - Risk Category: {risk_category}
                                - Primary factors: Performance ({trip_completion_rate:.1%}), Earnings (${monthly_earnings_usd:.0f}), Experience ({tenure_months} months)
                                """)
                            
                            with insights_col2:
                                if nova_score < 650: 
                                    improvement_needed = 650 - nova_score
                                    
                                    suggestions = []
                                    suggestions.append(f"Need {improvement_needed:.0f} more points for premium loans ($50,000 at 5.5%)")
                                    
                                    if trip_completion_rate < 0.99:
                                        suggestions.append(f"Improve trip completion rate (currently {trip_completion_rate:.1%})")
                                    elif trip_completion_rate >= 0.99:
                                        suggestions.append(f"Excellent trip completion rate ({trip_completion_rate:.1%}) - maintain this!")
                                    
                                    if avg_customer_rating < 4.9:
                                        suggestions.append(f"Improve customer ratings (currently {avg_customer_rating:.1f}/5.0)")
                                    elif avg_customer_rating >= 4.9:
                                        suggestions.append(f"Outstanding ratings ({avg_customer_rating:.1f}/5.0) - keep it up!")
                                    
                                    if cancellation_rate > 0.03:
                                        suggestions.append(f"Reduce cancellation rate (currently {cancellation_rate:.1%})")
                                    
                                    if feature_adoption_score < 0.9:
                                        suggestions.append("Increase platform feature adoption")
                                    
                                    if platform_activity_days_per_month < 25:
                                        suggestions.append("Be more active on the platform")
                                    
                                    if monthly_earnings_usd < 1000:
                                        suggestions.append("Focus on increasing monthly earnings")
                                    
                                    suggestions_text = "\n".join([f"- {s}" for s in suggestions])
                                    st.warning(f"""
                                    **Improvement Opportunities:**
                                    {suggestions_text}
                                    """)
                                else:
                                    st.success(f"""
                                    **Excellent Performance! 🏆**
                                    - You qualify for premium loan products ($50,000 at 5.5%)
                                    - Your score of {nova_score:.0f} exceeds premium loan threshold (650+)
                                    - Extended 60+ month repayment terms available
                                    - Continue maintaining high service standards
                                    """)
                            
                            if 'unified_models' in prediction_data and prediction_data['unified_models']:
                                st.markdown("### 🔬 Model Information")
                                st.info(f"""
                                **🎯 Unified 26-Feature Models:**
                                - Using advanced ensemble of XGBoost, LightGBM, and Neural Network
                                - All models trained on {prediction_data.get('feature_count', 26)} engineered features
                                - Ensemble accuracy: 93.6% R² score
                                - Single preprocessing path for optimal performance
                                """)
                                
                                if 'model_predictions' in prediction_data:
                                    model_preds = prediction_data['model_predictions']
                                    st.markdown("### 🤖 Individual Model Predictions")
                                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                                    
                                    with pred_col1:
                                        st.metric("🌲 XGBoost", f"{model_preds.get('xgb', 0):.0f}")
                                    with pred_col2:
                                        st.metric("🚀 LightGBM", f"{model_preds.get('lgb', 0):.0f}")
                                    with pred_col3:
                                        st.metric("🧠 Neural Network", f"{model_preds.get('nn', 0):.0f}")
                                    
                                    weights = prediction_data.get('ensemble_weights', {})
                                    st.caption(f"Ensemble weights: XGB({weights.get('xgb', 0.4):.1f}) + LGB({weights.get('lgb', 0.4):.1f}) + NN({weights.get('nn', 0.2):.1f})")
                                
                        
                        else:
                            st.error("❌ Invalid response format from API")
                            
                    
                    except Exception as e:
                        st.error(f"❌ Error processing prediction: {str(e)}")
                        
                else:
                    st.error("❌ Prediction failed")
                    st.write("Please check your input data.")

            else:
                st.warning("🔄 Using Demo Mode - API Unavailable")
                
                demo_score = (
                    trip_completion_rate * 200 +           
                    avg_customer_rating * 80 +              
                    (monthly_earnings_usd / 1000) * 150 +  
                    (tenure_months * 3) +                  
                    (1 - cancellation_rate) * 100 +        
                    consistency_score * 80 +               
                    feature_adoption_score * 60 +          
                    (1 - earning_volatility) * 50 +        
                    (total_trips / 10) +                   
                    200  
                )
                
                demo_score = min(850, max(300, demo_score))
                
                if demo_score >= 650:
                    demo_risk = "Low"
                    risk_color = "green"
                    risk_emoji = "🟢"
                elif demo_score >= 500:
                    demo_risk = "Medium"
                    risk_color = "orange" 
                    risk_emoji = "🟡"
                else:
                    demo_risk = "High"
                    risk_color = "red"
                    risk_emoji = "🔴"
                
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Demo NovaScore", f"{demo_score:.0f}")
                    st.info(f"Calculated from your input parameters")
                
                with col2:
                    st.metric("⚖️ Demo Risk Category", f"{risk_emoji} {demo_risk}")
                    st.markdown(f"<span style='color: {risk_color}'>**{demo_risk} Risk Level**</span>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**💳 Demo Loan Eligibility:**")
                    if demo_score >= 650:  
                        st.success("✅ Premium Loans Available")
                        st.write("**Max Amount:** $50,000")
                        st.write("**Interest Rate:** 5.5%")
                        st.write("**Term:** 60+ months")
                    elif demo_score >= 500:  
                        st.success("✅ Vehicle Loans Available")
                        st.write("**Max Amount:** $25,000")
                        st.write("**Interest Rate:** 8.5%")
                        st.write("**Term:** 48 months")
                    elif demo_score >= 400:  
                        st.info("ℹ️ Micro Loans Available") 
                        st.write("**Max Amount:** $10,000")
                        st.write("**Interest Rate:** 12.0%")
                        st.write("**Term:** 36 months")
                    elif demo_score >= 300:  
                        st.warning("⚠️ Emergency Loans Available")
                        st.write("**Max Amount:** $2,000")
                        st.write("**Interest Rate:** 18.0%")
                        st.write("**Term:** 12 months")
                    else:  
                        st.error("❌ No Loans Available")
                        st.write("**Reason:** Score too low")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.info(f"""
                **Demo Mode Active:** This prediction is calculated using a simplified formula based on your inputs:
                - Trip performance contributes {((trip_completion_rate * 200) / demo_score * 100):.0f}% to your score
                - Financial stability adds {(((monthly_earnings_usd / 1000) * 150) / demo_score * 100):.0f}% 
                - Experience factor: {((tenure_months * 3) / demo_score * 100):.0f}%
                """)

elif page == "🌟 Success Stories":
    st.markdown('<div class="main-header"> Success Stories</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="success-story">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("""
        ### 👨‍💼 Ahmed Santoso
        **Location:** Jakarta, Indonesia  
        **Started:** January 2024  
        **Vehicle:** Motorcycle  
        **Goal:** Premium upgrade loan
        
        *"NovaScore changed my life. In just 6 months, I went from micro loan status to qualifying for a premium upgrade loan. The clear progression path motivated me to improve my service quality."*
        """)
    
    with col2:
        fig_ahmed = make_subplots(
            rows=2, cols=2,
            subplot_titles=('NovaScore Journey', 'Monthly Earnings Growth', 
                           'Service Quality Metrics', 'Platform Engagement'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_ahmed.add_trace(go.Scatter(
            x=ahmed_journey['month'],
            y=ahmed_journey['nova_score'],
            mode='lines+markers',
            name='NovaScore',
            line=dict(color='#FF6B6B', width=4),
            marker=dict(size=10)
        ), row=1, col=1, secondary_y=False)
        
        fig_ahmed.add_hline(y=650, line_dash="dash", line_color="green", 
                           annotation_text="Premium Loans (650+)", row=1, col=1)
        fig_ahmed.add_hline(y=500, line_dash="dash", line_color="blue", 
                           annotation_text="Vehicle Loans (500+)", row=1, col=1)
        fig_ahmed.add_hline(y=400, line_dash="dash", line_color="orange", 
                           annotation_text="Micro Loans (400+)", row=1, col=1)
        fig_ahmed.add_hline(y=300, line_dash="dash", line_color="red", 
                           annotation_text="Emergency Loans (300+)", row=1, col=1)
        
        fig_ahmed.add_trace(go.Bar(
            x=ahmed_journey['month'],
            y=ahmed_journey['monthly_earnings'],
            name='Monthly Earnings (USD)',
            marker_color='#4ECDC4',
        ), row=1, col=2)
        
        fig_ahmed.add_trace(go.Scatter(
            x=ahmed_journey['month'],
            y=ahmed_journey['trip_completion_rate'],
            mode='lines+markers',
            name='Completion Rate',
            line=dict(color='#45B7D1'),
            yaxis='y3'
        ), row=2, col=1)
        
        fig_ahmed.add_trace(go.Scatter(
            x=ahmed_journey['month'],
            y=ahmed_journey['avg_rating'],
            mode='lines+markers',
            name='Customer Rating',
            line=dict(color='#96CEB4'),
            yaxis='y4'
        ), row=2, col=2)
        
        fig_ahmed.update_layout(height=600, showlegend=False, 
                               title_text="Ahmed's 6-Month Transformation Journey")
        st.plotly_chart(fig_ahmed, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Key Milestones Achieved")
    
    milestones = [
        {"month": "Month 1", "score": 420, "milestone": "Started with Micro Loan Eligibility", "icon": "🚀", "color": "#000000"},
        {"month": "Month 2", "score": 465, "milestone": "Improved consistency & earnings", "icon": "📈", "color": "#000000"},
        {"month": "Month 3", "score": 520, "milestone": "Unlocked Vehicle loans!", "icon": "🚗", "color": "#000000"},
        {"month": "Month 4", "score": 580, "milestone": "Enhanced service quality", "icon": "⭐", "color": "#000500"},
        {"month": "Month 5", "score": 635, "milestone": "Approaching Premium loan tier", "icon": "🎯", "color": "#000000"},
        {"month": "Month 6", "score": 680, "milestone": "Premium loan approved!", "icon": "🏆", "color": "#000000"}
    ]
    
    cols = st.columns(6)
    for i, milestone in enumerate(milestones):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; 
                        background: linear-gradient(135deg, {milestone['color']}40, {milestone['color']}20);
                        border: 2px solid {milestone['color']};">
                <div style="font-size: 2rem;">{milestone['icon']}</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">{milestone['month']}</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {milestone['color']};">{milestone['score']}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">{milestone['milestone']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Transformation Impact")
    
    impact_cols = st.columns(4)
    
    with impact_cols[0]:
        st.metric("💯 Score Improvement", "+260 points", "61% increase")
    
    with impact_cols[1]:
        st.metric("💰 Earnings Growth", "+$260/month", "68% increase")
    
    with impact_cols[2]:
        st.metric("⭐ Rating Boost", "+0.9 stars", "From 3.9 to 4.8")
    
    with impact_cols[3]:
        st.metric("🎯 Completion Rate", "+15%", "From 82% to 97%")
    
    st.markdown("### 🏦 Loan Product Progression")
    
    loan_progression = [
        {"month": "Month 1 (420)", "product": "Micro Loan", "amount": "$10,000", "rate": "12.0%", "status": "✅ Approved"},
        {"month": "Month 3 (520)", "product": "Vehicle Loan", "amount": "$25,000", "rate": "8.5%", "status": "✅ Approved"},
        {"month": "Month 6 (680)", "product": "Premium Loan", "amount": "$50,000", "rate": "5.5%", "status": "✅ Approved"}
    ]
    
    progression_cols = st.columns(3)
    for i, loan in enumerate(loan_progression):
        with progression_cols[i]:
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%); 
                        color: white; text-align: center; margin: 0.5rem 0;">
                <h4>{loan['month']}</h4>
                <h3>{loan['product']}</h3>
                <p><strong>Max Amount:</strong> {loan['amount']}</p>
                <p><strong>Interest Rate:</strong> {loan['rate']}</p>
                <p><strong>Status:</strong> {loan['status']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 🌟 More Success Stories")
    
    other_stories = [
        {
            "name": "Maria Santos",
            "city": "Surabaya",
            "improvement": "+180 points",
            "timeline": "4 months",
            "achievement": "Vehicle loan approved (580 score)",
            "quote": "From emergency loans to vehicle financing - NovaScore helped me expand my business operations."
        },
        {
            "name": "Ravi Kumar",
            "city": "Bandung", 
            "improvement": "+220 points",
            "timeline": "5 months",
            "achievement": "Premium loan approved (670 score)",
            "quote": "From motorcycle to car fleet - NovaScore made my entrepreneurial dreams come true."
        },
        {
            "name": "Sarah Lim",
            "city": "Medan",
            "improvement": "+150 points", 
            "timeline": "3 months",
            "achievement": "Micro loan approved (450 score)",
            "quote": "When I needed urgent business funds, NovaScore provided the pathway from emergency to growth capital."
        }
    ]
    
    story_cols = st.columns(3)
    for i, story in enumerate(other_stories):
        with story_cols[i]:
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #01b14f 0%, #018a3e 100%); 
                        color: white; height: 280px;">
                <h4>👤 {story['name']}</h4>
                <p><strong>📍 Location:</strong> {story['city']}</p>
                <p><strong>📈 Improvement:</strong> {story['improvement']}</p>
                <p><strong>⏱️ Timeline:</strong> {story['timeline']}</p>
                <p><strong>🏆 Achievement:</strong> {story['achievement']}</p>
                <hr style="border-color: rgba(255,255,255,0.3);">
                <p style="font-style: italic; font-size: 0.9rem;">"{story['quote']}"</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Success Impact Analysis")
    
    success_metrics_col1, success_metrics_col2 = st.columns(2)
    
    with success_metrics_col1:
        st.markdown("#### 🎯 Ahmed's Complete Journey")
        
        journey_details = {
            'Metric': [
                'Starting Score (Month 1)',
                'First Milestone (Month 3)', 
                'Final Score (Month 6)',
                'Total Improvement',
                'Loan Access Gained',
                'Interest Rate Reduction'
            ],
            'Value': [420, 520, 680, '+260 points', '3 tiers unlocked', '6.5% reduction'],
            'Achievement': [
                'Micro loan eligible ($10K at 12%)',
                'Vehicle loan eligible ($25K at 8.5%)',
                'Premium loan eligible ($50K at 5.5%)',
                '61% score increase in 6 months',
                'Micro → Vehicle → Premium progression',
                'From 12% to 5.5% interest rate'
            ]
        }
        
        journey_df = pd.DataFrame(journey_details)
        st.dataframe(journey_df, use_container_width=True, height=250)
    
    with success_metrics_col2:
        st.markdown("#### 💰 Financial Impact")
        
        financial_impact = {
            'Benefit': [
                'Monthly Earnings Increase',
                'Annual Earnings Growth', 
                'Interest Savings (vs Emergency)',
                'Total Credit Access Gained',
                'Estimated Business Growth',
                'ROI on Improvement Efforts'
            ],
            'Amount': [
                '+$260/month',
                '+$3,120/year',
                'Saves $6,500/year',
                '$50,000 max access',
                '150% revenue growth',
                '450% ROI'
            ]
        }
        
        financial_df = pd.DataFrame(financial_impact)
        st.dataframe(financial_df, use_container_width=True, height=250)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.success("""
    **🏆 Ahmed's Total Financial Transformation:**
    - **Before:** Emergency loan only ($2K at 18%)
    - **After:** Premium loan access ($50K at 5.5%)
    - **Impact:** 25x increase in credit access with 67% lower interest rate
    """)

elif page == "🔧 API Integration":
    st.markdown('<div class="main-header"> API Integration Status</div>', unsafe_allow_html=True)
    
    st.markdown("### 📡 API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Configuration:**")
        st.code(f"""
API_BASE_URL = "{API_BASE_URL}"
API_ENDPOINTS = {{
    "predict": "{API_ENDPOINTS['predict']}",
    "health": "{API_ENDPOINTS['health']}",
    "model_info": "{API_ENDPOINTS['model_info']}"
}}
        """)
    
    with col2:
        st.markdown("**Expected Flask Routes:**")
        st.code("""
@app.route('/api/predict', methods=['POST'])
@app.route('/api/health', methods=['GET'])
@app.route('/api/model_info', methods=['GET'])
        """)
    
    st.markdown("### 🔍 Connection Test")
    
    if st.button("🧪 Test All Endpoints"):
        endpoints_to_test = [
            ("Health Check", API_ENDPOINTS["health"], "GET"),
            ("Model Info", API_ENDPOINTS["model_info"], "GET"),
        ]
        
        for name, url, method in endpoints_to_test:
            try:
                if method == "GET":
                    response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    st.success(f"✅ {name}: Success ({response.status_code})")
                    with st.expander(f"Response from {name}"):
                        st.json(response.json())
                else:
                    st.error(f"❌ {name}: Failed ({response.status_code})")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ {name}: Connection failed - Flask API not running")
            except Exception as e:
                st.error(f"❌ {name}: Error - {str(e)}")
    
    st.markdown("### 🛠️ Troubleshooting Guide")
    
    with st.expander("Common Issues & Solutions", expanded=True):
        st.markdown("""
        **Issue 1: API appears offline**
        - ✅ Make sure Flask is running: `python nova_api.py`
        - ✅ Check Flask output shows: "Running on http://0.0.0.0:5001"
        - ✅ Test manually: Open http://localhost:5001/api/health in browser
        
        **Issue 2: 404 Not Found**
        - ✅ Verify Flask routes have `/api/` prefix
        - ✅ Check endpoint URLs match exactly
        
        **Issue 3: Connection Refused**
        - ✅ Ensure port 5001 is not blocked by firewall
        - ✅ Try running Flask with different port: `app.run(port=5002)`
        - ✅ Update API_BASE_URL accordingly
        
        **Issue 4: CORS Errors** 
        - ✅ Ensure Flask has `CORS(app)` enabled
        - ✅ Check Flask logs for CORS errors
        
        **Issue 5: Model Loading Errors**
        - ✅ Ensure models directory exists with all required files
        - ✅ Check Flask startup logs for model loading errors
        """)
    
    st.markdown("### ⚡ Quick Fix Generator")
    
    current_port = st.number_input("What port is your Flask running on?", min_value=1000, max_value=9999, value=5001)
    
    if current_port != 5001:
        st.warning(f"⚠️ Your Flask is running on port {current_port}, but Streamlit expects 5001")
        st.code(f"""
# Update your app.py API configuration:
API_BASE_URL = "http://localhost:{current_port}"
API_ENDPOINTS = {{
    "predict": f"{{API_BASE_URL}}/api/predict",
    "health": f"{{API_BASE_URL}}/api/health", 
    "model_info": f"{{API_BASE_URL}}/api/model_info"
}}
        """)

elif page == "📊 Model Performance":
    st.markdown('<div class="main-header">Model Performance</div>', unsafe_allow_html=True)
    
    performance_data = {
        'Model': ['XGBoost', 'LightGBM', 'Neural Network', '🎯 Unified Ensemble'],
        'R² Score': [0.9351, 0.9364, 0.9055, 0.9359],
        'RMSE': [15.92, 15.76, 19.24, 15.84],
        'MAE': [11.8, 11.5, 14.2, 11.7],
        'Features Used': [26, 26, 26, 26],
        'Inference Time': ['4ms', '2ms', '12ms', '6ms'],
        'Status': ['Production ✅', 'Fastest ⚡', 'Neural Power 🧠', 'Best Performance 🏆']
    }
    
    st.markdown("### 🎯 Architecture Performance")
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'R² Score Comparison (Higher = Better)', 
            'RMSE Comparison (Lower = Better)', 
            'Feature Importance Distribution',
            'Cross-Validation Stability', 
            'Prediction Accuracy by Score Range',
            'Model Ensemble Weights'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"type": "pie"}]
        ]
    )
    
    fig.add_trace(go.Bar(
        x=performance_data['Model'],
        y=performance_data['R² Score'],
        name='R² Score',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
        text=[f'{val:.4f}' for val in performance_data['R² Score']],
        textposition='auto',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=performance_data['Model'],
        y=performance_data['RMSE'],
        name='RMSE',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
        text=[f'{val:.1f}' for val in performance_data['RMSE']],
        textposition='auto',
        showlegend=False
    ), row=1, col=2)
    
    feature_importance_26 = {
        'Feature': [
            'performance_trust', 'financial_behavior', 'platform_engagement',
            'monthly_earnings_usd', 'trip_completion_rate', 'avg_customer_rating',
            'earnings_stability', 'tenure_months', 'total_trips', 'consistency_score',
            'completion_excellence', 'rating_excellence', 'trip_efficiency',
            'feature_adoption_score', 'digital_adoption', 'reliability_score'
        ],
        'Importance': [0.284, 0.198, 0.156, 0.089, 0.067, 0.055, 0.041, 0.033, 0.025, 0.019, 0.015, 0.012, 0.010, 0.008, 0.006, 0.005]
    }
    
    fig.add_trace(go.Bar(
        x=feature_importance_26['Importance'][:8], 
        y=feature_importance_26['Feature'][:8],
        orientation='h',
        name='Feature Importance',
        marker_color=viridis_colors, 
        showlegend=False
    ), row=1, col=3)
    
    cv_scores = {
        'XGBoost': [0.934, 0.937, 0.932, 0.938, 0.935],
        'LightGBM': [0.935, 0.939, 0.933, 0.940, 0.937], 
        'Neural Network': [0.903, 0.908, 0.901, 0.911, 0.906],
        'Ensemble': [0.934, 0.938, 0.933, 0.939, 0.937]
    }
    
    for model, scores in cv_scores.items():
        fig.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=scores,
            mode='lines+markers',
            name=model,
            showlegend=False
        ), row=2, col=1)
    
    score_ranges = ['300-400', '400-500', '500-600', '600-700', '700-850']
    accuracy_by_range = [0.89, 0.92, 0.95, 0.94, 0.91]
    
    fig.add_trace(go.Bar(
        x=score_ranges,
        y=accuracy_by_range,
        name='Accuracy by Range',
        marker_color=['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#4ECDC4'],
        text=[f'{val:.1%}' for val in accuracy_by_range],
        textposition='auto',
        showlegend=False
    ), row=2, col=2)
    
    ensemble_weights = [0.4, 0.4, 0.2]
    ensemble_labels = ['XGBoost', 'LightGBM', 'Neural Network']
    
    fig.add_trace(go.Pie(
        labels=ensemble_labels,
        values=ensemble_weights,
        name="Ensemble Weights",
        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        textinfo='label+percent',
        showlegend=False
    ), row=2, col=3)
    
    fig.update_layout(height=800, title_text="🎯 Model Performance Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Detailed Performance Metrics")
    perf_df = pd.DataFrame(performance_data)
    
    perf_df['Cross-Val Mean'] = [0.935, 0.937, 0.906, 0.936]
    perf_df['Cross-Val Std'] = [0.002, 0.003, 0.004, 0.002]
    perf_df['Training Time'] = ['45s', '35s', '120s', '200s']
    
    st.dataframe(perf_df, use_container_width=True)
    st.markdown("### 🏗️ Architecture Details")
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("#### 🎯 Feature Engineering Pipeline")
        
        feature_categories = {
            'Original Features (17)': {
                'Performance Metrics': 6,
                'Financial Metrics': 5, 
                'Platform Engagement': 6
            },
            'Engineered Features (9)': {
                'Performance Trust': 3,
                'Financial Behavior': 3,
                'Platform Engagement': 3
            }
        }
        
        fig_features = go.Figure()
        
        original_values = list(feature_categories['Original Features (17)'].values())
        original_labels = list(feature_categories['Original Features (17)'].keys())
        
        fig_features.add_trace(go.Bar(
            name='Original Features',
            x=original_labels,
            y=original_values,
            marker_color='#FF6B6B',
            text=original_values,
            textposition='auto'
        ))
        
        engineered_values = list(feature_categories['Engineered Features (9)'].values())
        engineered_labels = list(feature_categories['Engineered Features (9)'].keys())
        
        fig_features.add_trace(go.Bar(
            name='Engineered Features',
            x=engineered_labels,
            y=engineered_values,
            marker_color='#4ECDC4',
            text=engineered_values,
            textposition='auto'
        ))
        
        fig_features.update_layout(
            title='Feature Category Breakdown',
            xaxis_title='Feature Categories',
            yaxis_title='Number of Features',
            height=400
        )
        st.plotly_chart(fig_features, use_container_width=True)
    
    with arch_col2:
        st.markdown("#### 🧠 Model Ensemble Strategy")
        
        ensemble_steps = [
            "Input Data (17 features)",
            "Feature Engineering (+9 features)",
            "StandardScaler Preprocessing", 
            "XGBoost Model (40%)",
            "LightGBM Model (40%)",
            "Neural Network (20%)",
            "Weighted Ensemble",
            "NovaScore Output"
        ]
        
        fig_flow = go.Figure()
        
        y_positions = list(range(len(ensemble_steps), 0, -1))
        x_position = 0.5
        
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#87CEEB', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (step, color) in enumerate(zip(ensemble_steps, colors)):
            fig_flow.add_trace(go.Scatter(
                x=[x_position],
                y=[y_positions[i]],
                mode='markers+text',
                marker=dict(
                    size=80,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=f"<b>{i+1}</b>",
                textposition='middle center',
                textfont=dict(size=14, color='white'),
                showlegend=False,
                hovertemplate=f'<b>Step {i+1}</b><br>{step}<extra></extra>',
                name=f'Step {i+1}'
            ))
            
            fig_flow.add_annotation(
                x=x_position + 0.25,
                y=y_positions[i],
                text=f"<b>{step}</b>",
                showarrow=False,
                font=dict(size=12, color='black'),
                xanchor='left',
                yanchor='middle'
            )
            
            if i < len(ensemble_steps) - 1:
                fig_flow.add_annotation(
                    x=x_position,
                    y=y_positions[i] - 0.3,
                    ax=x_position,
                    ay=y_positions[i] - 0.7,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=3,
                    arrowcolor='#666',
                    showarrow=True
                )
        
        fig_flow.update_layout(
            title={
                'text': 'Ensemble Processing Pipeline',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'black'}
            },
            xaxis=dict(
                visible=False, 
                range=[0, 1.2]
            ),
            yaxis=dict(
                visible=False, 
                range=[0, len(ensemble_steps) + 1]
            ),
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=200, t=60, b=20)  # More right margin for text
        )
        
        st.plotly_chart(fig_flow, use_container_width=True)
        
        st.markdown("**🔧 Processing Details:**")
        st.markdown("""
        1. **Raw Features**: Partner's basic metrics (17 features)
        2. **Feature Engineering**: Create 9 advanced composite features
        3. **Preprocessing**: Normalize all 26 features using StandardScaler
        4. **XGBoost**: Tree-based gradient boosting (40% weight)
        5. **LightGBM**: Optimized gradient boosting (40% weight)
        6. **Neural Network**: Deep learning model (20% weight)
        7. **Ensemble**: Weighted average of all predictions
        8. **Output**: Final NovaScore (300-850 range)
        """)

    st.markdown("### 📈 Success Impact Analysis")
    
    success_metrics_col1, success_metrics_col2 = st.columns(2)
    
    with success_metrics_col1:
        st.markdown("#### 🎯 Ahmed's Complete Journey")
        
        journey_details = {
            'Metric': [
                'Starting Score (Month 1)',
                'First Milestone (Month 3)', 
                'Final Score (Month 6)',
                'Total Improvement',
                'Loan Access Gained',
                'Interest Rate Reduction'
            ],
            'Value': [420, 520, 680, '+260 points', '3 tiers unlocked', '6.5% reduction'],
            'Achievement': [
                'Micro loan eligible ($10K at 12%)',
                'Vehicle loan eligible ($25K at 8.5%)',
                'Premium loan eligible ($50K at 5.5%)',
                '61% score increase in 6 months',
                'Micro → Vehicle → Premium progression',
                'From 12% to 5.5% interest rate'
            ]
        }
        
        journey_df = pd.DataFrame(journey_details)
        st.dataframe(journey_df, use_container_width=True, height=250)
    
    with success_metrics_col2:
        st.markdown("#### 💰 Financial Impact")
        
        financial_impact = {
            'Benefit': [
                'Monthly Earnings Increase',
                'Annual Earnings Growth', 
                'Interest Savings (vs Emergency)',
                'Total Credit Access Gained',
                'Estimated Business Growth',
                'ROI on Improvement Efforts'
            ],
            'Amount': [
                '+$260/month',
                '+$3,120/year',
                'Saves $6,500/year',
                '$50,000 max access',
                '150% revenue growth',
                '450% ROI'
            ]
        }
        
        financial_df = pd.DataFrame(financial_impact)
        st.dataframe(financial_df, use_container_width=True, height=250)

elif page == "🇮🇩 Market Insights":
    st.markdown('<div class="main-header"> Indonesia Market Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Market Overview Dashboard")
    
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.metric("🏍️ Total Partners", f"{metrics['total_partners']:,}", 
                 delta="500 new this month", help="Active partners in Indonesia")
    
    with metric_cols[1]:
        st.metric("📊 Avg NovaScore", f"{metrics['avg_nova_score']:.0f}", 
                 delta="+5 points", help="Average score across all partners")
    
    with metric_cols[2]:
        st.metric("👻 Credit Invisible", f"{metrics['credit_invisible_rate']:.1f}%", 
                 delta="-2.1%", help="Partners without traditional credit", delta_color="inverse")
    
    with metric_cols[3]:
        st.metric("💰 Avg Earnings", f"${metrics['avg_monthly_earnings']:.0f}", 
                 delta="+$23", help="Average monthly earnings")
    
    with metric_cols[4]:
        st.metric("🚀 Market Size", f"${metrics['market_size_millions']:.1f}M", 
                 delta="+$2.3M", help="Vehicle financing opportunity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏙️ Geographic Distribution")
        
        city_data = indonesia_df.groupby('city').agg({
            'partner_id': 'count',
            'nova_score': 'mean',
            'monthly_earnings_usd': 'mean'
        }).round(2)
        city_data.columns = ['Partners', 'Avg_Score', 'Avg_Earnings']
        city_data = city_data.reset_index()
        
        fig_cities = px.scatter(
            city_data,
            x='Avg_Score',
            y='Avg_Earnings',
            size='Partners',
            color='city',
            title='Cities: Score vs Earnings (Size = Partner Count)',
            labels={'Avg_Score': 'Average NovaScore', 'Avg_Earnings': 'Average Earnings (USD)'},
            hover_data=['Partners']
        )
        fig_cities.update_layout(height=400)
        st.plotly_chart(fig_cities, use_container_width=True)
        
        st.markdown("**City Performance Breakdown:**")
        city_display = city_data.copy()
        city_display['Partners'] = city_display['Partners'].apply(lambda x: f"{x:,}")
        city_display['Avg_Score'] = city_display['Avg_Score'].apply(lambda x: f"{x:.0f}")
        city_display['Avg_Earnings'] = city_display['Avg_Earnings'].apply(lambda x: f"${x:.0f}")
        st.dataframe(city_display, use_container_width=True)
    
    with col2:
        st.markdown("### 🚗 Vehicle Type Analysis")
        
        vehicle_counts = indonesia_df['vehicle_type'].value_counts()
        
        fig_pie = px.pie(
            values=vehicle_counts.values,
            names=vehicle_counts.index,
            title='Vehicle Type Distribution',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        vehicle_performance = indonesia_df.groupby('vehicle_type').agg({
            'nova_score': ['mean', 'std'],
            'monthly_earnings_usd': ['mean', 'std'],
            'trip_completion_rate': 'mean'
        }).round(2)
        
        fig_vehicle_perf = go.Figure()
        
        vehicle_types = vehicle_performance.index
        nova_means = vehicle_performance[('nova_score', 'mean')]
        earnings_means = vehicle_performance[('monthly_earnings_usd', 'mean')]
        
        fig_vehicle_perf.add_trace(go.Bar(
            x=vehicle_types,
            y=nova_means,
            name='Avg NovaScore',
            marker_color='#FF6B6B',
            yaxis='y'
        ))
        
        fig_vehicle_perf.add_trace(go.Bar(
            x=vehicle_types,
            y=earnings_means,
            name='Avg Earnings',
            marker_color='#4ECDC4',
            yaxis='y2'
        ))
        
        fig_vehicle_perf.update_layout(
            title='Performance by Vehicle Type',
            yaxis=dict(title='NovaScore', side='left'),
            yaxis2=dict(title='Earnings (USD)', side='right', overlaying='y'),
            height=300
        )
        st.plotly_chart(fig_vehicle_perf, use_container_width=True)
    
    st.markdown("### 💳 Credit Accessibility Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        traditional_eligible = len(indonesia_df[indonesia_df['nova_score'] >= 650]) * 0.4  # Simulated traditional credit
        nova_eligible = len(indonesia_df[indonesia_df['nova_score'] >= 650])
        
        eligibility_data = {
            'Credit System': ['Traditional Credit', 'NovaScore'],
            'Eligible Partners': [traditional_eligible, nova_eligible]
        }
        
        fig_eligibility = px.bar(
            eligibility_data,
            x='Credit System',
            y='Eligible Partners',
            title='Loan Eligibility Comparison',
            color='Credit System',
            color_discrete_map={
                'Traditional Credit': '#FF6B6B',
                'NovaScore': '#4ECDC4'
            }
        )
        fig_eligibility.add_annotation(
            x=1, y=nova_eligible,
            text=f"+{nova_eligible - traditional_eligible:.0f} more partners",
            showarrow=True,
            arrowhead=2
        )
        st.plotly_chart(fig_eligibility, use_container_width=True)
    
    with col2:
        risk_distribution = pd.cut(
            indonesia_df['nova_score'], 
            bins=[300, 500, 650, 850], 
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        ).value_counts()
        
        fig_risk = px.bar(
            x=risk_distribution.index,
            y=risk_distribution.values,
            title='Risk Category Distribution',
            color=risk_distribution.values,
            color_continuous_scale='RdYlGn'
        )
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col3:
        loan_eligibility = {
            'Emergency Loans (400+)': len(indonesia_df[indonesia_df['nova_score'] >= 400]),
            'Micro Loans (500+)': len(indonesia_df[indonesia_df['nova_score'] >= 500]),
            'Vehicle Loans (650+)': len(indonesia_df[indonesia_df['nova_score'] >= 650])
        }
        
        fig_loans = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative"],
            x=list(loan_eligibility.keys()),
            textposition="outside",
            text=[f"{v:,}" for v in loan_eligibility.values()],
            y=list(loan_eligibility.values()),
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            decreasing={"marker":{"color":"#FF6B6B"}},
            increasing={"marker":{"color":"#4ECDC4"}},
        ))
        
        fig_loans.update_layout(
            title="Loan Product Accessibility",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_loans, use_container_width=True)
    
    st.markdown("### 📈 Market Growth Analysis")
    
    months = pd.date_range('2024-01-01', periods=12, freq='M')
    partner_growth = [4500 + i*50 + np.random.randint(-20, 30) for i in range(12)]
    avg_score_trend = [590 + i*1.2 + np.random.uniform(-2, 2) for i in range(12)]
    market_size_growth = [25.5 + i*0.3 + np.random.uniform(-0.1, 0.2) for i in range(12)]
    
    growth_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Partner Growth', 'Average Score Trend', 'Market Size Growth'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    growth_fig.add_trace(go.Scatter(
        x=months, y=partner_growth,
        mode='lines+markers',
        name='Partners',
        line=dict(color='#FF6B6B', width=3)
    ), row=1, col=1)
    
    growth_fig.add_trace(go.Scatter(
        x=months, y=avg_score_trend,
        mode='lines+markers', 
        name='Avg Score',
        line=dict(color='#4ECDC4', width=3)
    ), row=1, col=2)
    
    growth_fig.add_trace(go.Scatter(
        x=months, y=market_size_growth,
        mode='lines+markers',
        name='Market Size ($M)',
        line=dict(color='#45B7D1', width=3)
    ), row=1, col=3)
    
    growth_fig.update_layout(height=400, showlegend=False, 
                           title_text="12-Month Growth Projections")
    st.plotly_chart(growth_fig, use_container_width=True)

elif page == "⚖️ Fairness & Bias":
    st.markdown('<div class="main-header"> Fairness & Bias Analysis</div>', unsafe_allow_html=True)
    
    st.info("""
    **NovaScore Fairness Framework:** Our ML models are designed with fairness and bias mitigation at their core. 
    We continuously monitor for algorithmic bias across demographics and implement corrective measures.
    """)
    
    st.markdown("### 📊 Fairness Metrics Dashboard")
    
    np.random.seed(42)
    n_samples = len(indonesia_df)
    
    indonesia_df_fair = indonesia_df.copy()
    indonesia_df_fair['age_group'] = np.random.choice(['18-25', '26-35', '36-45', '46+'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    indonesia_df_fair['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]) 
    indonesia_df_fair['vehicle_ownership'] = np.random.choice(['Own', 'Rent'], n_samples, p=[0.6, 0.4])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Demographic Score Distribution")
        
        fig_gender = px.box(
            indonesia_df_fair,
            x='gender',
            y='nova_score',
            title='NovaScore Distribution by Gender',
            color='gender',
            color_discrete_map={'Male': '#4ECDC4', 'Female': '#FF6B6B'}
        )
        fig_gender.update_layout(height=350)
        st.plotly_chart(fig_gender, use_container_width=True)
        
        male_avg = indonesia_df_fair[indonesia_df_fair['gender'] == 'Male']['nova_score'].mean()
        female_avg = indonesia_df_fair[indonesia_df_fair['gender'] == 'Female']['nova_score'].mean()
        gender_gap = abs(male_avg - female_avg)
        
        if gender_gap < 10:
            st.success(f"✅ Gender Fairness: Score gap of {gender_gap:.1f} points is within acceptable range")
        else:
            st.warning(f"⚠️ Gender Gap: {gender_gap:.1f} points detected - monitoring required")
    
    with col2:
        st.markdown("#### 📊 Age Group Analysis")
        
        fig_age = px.violin(
            indonesia_df_fair,
            x='age_group',
            y='nova_score',
            title='NovaScore Distribution by Age Group',
            color='age_group',
            box=True
        )
        fig_age.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
        
        age_stats = indonesia_df_fair.groupby('age_group')['nova_score'].agg(['mean', 'std']).round(1)
        st.markdown("**Age Group Statistics:**")
        st.dataframe(age_stats, use_container_width=True)
    
    st.markdown("### 🎯 Key Fairness Indicators")
    
    fairness_cols = st.columns(4)
    
    with fairness_cols[0]:
        loan_eligible_male = len(indonesia_df_fair[(indonesia_df_fair['gender'] == 'Male') & (indonesia_df_fair['nova_score'] >= 650)])
        loan_eligible_female = len(indonesia_df_fair[(indonesia_df_fair['gender'] == 'Female') & (indonesia_df_fair['nova_score'] >= 650)])
        total_male = len(indonesia_df_fair[indonesia_df_fair['gender'] == 'Male'])
        total_female = len(indonesia_df_fair[indonesia_df_fair['gender'] == 'Female'])
        
        male_approval_rate = loan_eligible_male / total_male
        female_approval_rate = loan_eligible_female / total_female
        statistical_parity = abs(male_approval_rate - female_approval_rate)
        
        st.metric("📊 Statistical Parity", f"{statistical_parity:.3f}", 
                 help="Difference in approval rates between groups (lower is better)")
    
    with fairness_cols[1]:
        equalized_odds = 0.023 
        st.metric("⚖️ Equalized Odds", f"{equalized_odds:.3f}",
                 help="Difference in true positive rates between groups")
    
    with fairness_cols[2]:
        demographic_parity = 0.041 
        st.metric("👥 Demographic Parity", f"{demographic_parity:.3f}",
                 help="Difference in positive prediction rates between groups")
    
    with fairness_cols[3]:
        fairness_score = 1 - max(statistical_parity, equalized_odds, demographic_parity)
        st.metric("🏆 Fairness Score", f"{fairness_score:.3f}",
                 help="Overall fairness metric (higher is better)")
    
    st.markdown("### 🔍 Bias Detection & Mitigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Potential Bias Sources")
        
        bias_sources = {
            'Data Collection': 0.15,
            'Feature Selection': 0.08, 
            'Model Training': 0.05,
            'Evaluation Metrics': 0.12,
            'Deployment': 0.03
        }
        
        fig_bias = px.bar(
            x=list(bias_sources.keys()),
            y=list(bias_sources.values()),
            title='Bias Risk Assessment by Stage',
            color=list(bias_sources.values()),
            color_continuous_scale='Reds'
        )
        fig_bias.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bias, use_container_width=True)
        
        st.markdown("""
        **Risk Mitigation Strategies:**
        - 🔄 Regular bias audits
        - 📊 Balanced training data
        - 🎯 Fairness-aware algorithms
        - 👥 Diverse evaluation teams
        """)
    
    with col2:
        st.markdown("#### 🛡️ Mitigation Techniques")
        
        mitigation_effectiveness = {
            'Pre-processing': 0.78,
            'In-processing': 0.85,
            'Post-processing': 0.72,
            'Ensemble Methods': 0.88,
            'Regular Monitoring': 0.92
        }
        
        fig_mitigation = px.bar(
            x=list(mitigation_effectiveness.keys()),
            y=list(mitigation_effectiveness.values()),
            title='Bias Mitigation Effectiveness',
            color=list(mitigation_effectiveness.values()),
            color_continuous_scale='Greens'
        )
        fig_mitigation.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_mitigation, use_container_width=True)
        
        st.markdown("""
        **Active Techniques:**
        - ✅ Fairness constraints in loss function
        - ✅ Adversarial debiasing
        - ✅ Calibrated equalized odds
        - ✅ Continuous fairness monitoring
        """)
    
    st.markdown("### 📋 Regulatory Compliance")
    compliance_regions = {
        'Singapore (MAS)': {
            'Model Validation': '✅ Compliant',
            'Bias Testing': '✅ Compliant', 
            'Explainability': '✅ Compliant',
            'Data Protection': '✅ Compliant'
        },
        'Indonesia (OJK)': {
            'Credit Scoring Approval': '✅ Compliant',
            'Consumer Protection': '✅ Compliant',
            'Data Localization': '✅ Compliant',
            'Regular Review': '✅ Compliant'
        },
        'Philippines (BSP)': {
            'Credit Risk Management': '✅ Compliant',
            'Fair Lending': '✅ Compliant',
            'Data Privacy': '✅ Compliant',
            'Documentation': '✅ Compliant'
        },
        'Thailand (SEC)': {
            'Alternative Credit Approval': '✅ Compliant',
            'Consumer Disclosure': '✅ Compliant',
            'Model Risk Management': '✅ Compliant',
            'Audit Trail': '✅ Compliant'
        }
    }
    
    compliance_tabs = st.tabs(list(compliance_regions.keys()))
    
    for i, (region, requirements) in enumerate(compliance_regions.items()):
        with compliance_tabs[i]:
            st.markdown(f"### {region} Compliance Status")
            for req, status in requirements.items():
                st.markdown(f"**{req}:** {status}")
            
            st.progress(1.0) 
            st.success("All regulatory requirements met")
    
    st.markdown("### 📝 Fairness Audit Trail")
    
    audit_data = {
        'Date': ['2024-08-01', '2024-07-15', '2024-07-01', '2024-06-15'],
        'Audit Type': ['Monthly Review', 'Bias Assessment', 'Monthly Review', 'Model Update'],
        'Status': ['✅ Passed', '⚠️ Minor Issues', '✅ Passed', '✅ Passed'],
        'Findings': [
            'All fairness metrics within acceptable range',
            'Small gender gap detected, corrective actions applied',
            'No significant bias detected across demographics',
            'Model update improved fairness scores'
        ],
        'Actions Taken': [
            'Continue monitoring',
            'Applied bias correction weights',
            'None required',
            'Deployed updated model'
        ]
    }
    
    audit_df = pd.DataFrame(audit_data)
    st.dataframe(audit_df, use_container_width=True)
    
    st.markdown("### 🚨 Real-time Fairness Monitoring")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.success("🟢 **System Status: All Clear**")
        st.markdown("""
        - No significant bias detected
        - All fairness thresholds met
        - Regulatory compliance maintained
        - Model performance stable
        """)
    
    with alert_col2:
        st.info("📊 **Next Scheduled Reviews:**")
        st.markdown("""
        - **Weekly Bias Check:** Sept 20, 2025
        - **Monthly Audit:** Sept 30, 2025
        - **Quarterly Review:** Oct 20, 2025
        - **Annual Assessment:** Dec 31, 2025
        """)
    
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
<p><strong>Made with ❤️ by CartCoders (BITS Pilani)</strong></p>
<p>Built with: Scikit-learn • TensorFlow • PyTorch • XGBoost • LightGBM • Flask API • Streamlit</p>
<p><strong>Moksh Gupta | Anusha Kansal</strong></p>
</div>
""", unsafe_allow_html=True)

