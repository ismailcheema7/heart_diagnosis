import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import io
from lightgbm import LGBMClassifier
import category_encoders as ce
from imblearn.ensemble import EasyEnsembleClassifier
import shap
import plotly.express as px
import plotly.graph_objects as go

# Load the pickled model and encoder
with open('best_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

with open('cbe_encoder.pkl', 'rb') as encoder_file:
    encoder = pkl.load(encoder_file)

# Load the dataset for reference
data = pd.read_csv('brfss2022_data_wrangling_output.zip', compression='zip')
data['heart_disease'] = data['heart_disease'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')

# Page config with HoloMed AI branding
st.set_page_config(
    layout='wide', 
    page_title='HoloMed AI - Heart Disease Assessment', 
    page_icon="💙",
    initial_sidebar_state="collapsed"
)

# Custom CSS with HoloMed AI styling
def local_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1e3a8a 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .holomedai-header {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.2);
    }
    
    .holomedai-title {
        background: linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(6, 182, 212, 0.5);
    }
    
    .holomedai-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .holomedai-mission {
        color: #06b6d4;
        font-size: 1rem;
        margin-top: 1rem;
        font-style: italic;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.1);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-color: rgba(6, 182, 212, 0.6);
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.2);
        transform: translateY(-2px);
    }
    
    .info-card h3 {
        color: #06b6d4;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .info-card h3::before {
        content: "⚡";
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    .info-card p, .info-card li {
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .info-card ul {
        padding-left: 1.2rem;
    }
    
    /* Form Sections */
    .form-section {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.1);
    }
    
    .form-section h4 {
        color: #06b6d4;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(6, 182, 212, 0.3);
    }
    
    /* Streamlit Input Styling */
    .stSelectbox > div > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 1px #06b6d4 !important;
    }
    
    .stSelectbox label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    /* Custom Button */
    .assessment-button {
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.3);
        width: 100%;
        margin: 1rem 0;
    }
    
    .assessment-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.4);
    }
    
    /* Results Section */
    .results-card {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.2);
    }
    
    .risk-score {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    
    .risk-low { color: #10b981; }
    .risk-moderate { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    
    /* Recommendations */
    .recommendation {
        background: rgba(6, 182, 212, 0.1);
        border-left: 4px solid #06b6d4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
    }
    
    /* Footer */
    .holomedai-footer {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
    }
    
    .holomedai-footer h4 {
        color: #06b6d4;
        margin-bottom: 1rem;
    }
    
    .social-links a {
        color: #06b6d4;
        text-decoration: none;
        margin: 0 1rem;
        transition: color 0.3s ease;
    }
    
    .social-links a:hover {
        color: #3b82f6;
    }
    
    /* Contact Form */
    .contact-form {
        background: rgba(15, 23, 42, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    
    .contact-form input, .contact-form textarea {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        padding: 0.75rem !important;
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    
    .contact-form input:focus, .contact-form textarea:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 1px #06b6d4 !important;
    }
    
    .contact-form button {
        background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        color: white !important;
        font-weight: 600 !important;
        cursor: pointer !important;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        color: #fca5a5;
    }
    
    .tech-badge {
        display: inline-block;
        background: rgba(6, 182, 212, 0.2);
        border: 1px solid rgba(6, 182, 212, 0.4);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        color: #06b6d4;
        margin: 0.2rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Header Section
st.markdown("""
<div class="holomedai-header">
    <h1 class="holomedai-title">HoloMed AI</h1>
    <p class="holomedai-subtitle">AI-Powered Heart Disease Risk Assessment</p>
    <p class="holomedai-mission">Providing accessible education on the transformative impact of AI in Medicine</p>
</div>
""", unsafe_allow_html=True)

# Technology badges
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <span class="tech-badge">🤖 Machine Learning</span>
    <span class="tech-badge">📊 SHAP Analysis</span>
    <span class="tech-badge">🧠 LightGBM</span>
    <span class="tech-badge">⚡ Real-time Prediction</span>
    <span class="tech-badge">🔬 Evidence-Based</span>
</div>
""", unsafe_allow_html=True)

# Info Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>Our Mission</h3>
        <p>HoloMed AI leverages cutting-edge artificial intelligence to provide personalized cardiovascular risk assessments. Our advanced models analyze multiple health factors to deliver actionable insights that empower you to take control of your heart health.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>How It Works</h3>
        <ul>
            <li><strong>Data Input:</strong> Enter your comprehensive health profile</li>
            <li><strong>AI Analysis:</strong> Advanced algorithms process your information</li>
            <li><strong>Risk Calculation:</strong> Receive your personalized risk score</li>
            <li><strong>Smart Recommendations:</strong> Get AI-powered actionable advice</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# User Input Section
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown("#### 👤 Demographics")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["female", "male", "nonbinary"], index=1)
with col2:
    race = st.selectbox("Race/Ethnicity", [
        "white_only_non_hispanic", "black_only_non_hispanic", "asian_only_non_hispanic", 
        "american_indian_or_alaskan_native_only_non_hispanic", "multiracial_non_hispanic", 
        "hispanic", "native_hawaiian_or_other_pacific_islander_only_non_hispanic"
    ], index=0)
with col3:
    age_category = st.selectbox("Age Group", [
        "Age_18_to_24", "Age_25_to_29", "Age_30_to_34", "Age_35_to_39", 
        "Age_40_to_44", "Age_45_to_49", "Age_50_to_54", "Age_55_to_59",
        "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79",
        "Age_80_or_older"
    ], index=4)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown("#### 🏥 Medical History")

col1, col2, col3 = st.columns(3)

with col1:
    general_health = st.selectbox("Overall Health Rating", ["excellent", "very_good", "good", "fair", "poor"], index=0)
    heart_attack = st.selectbox("History of Heart Attack", ["yes", "no"], index=1, help="Heart attack occurs when blood flow to the heart is blocked")
    kidney_disease = st.selectbox("Kidney Disease Diagnosis", ["yes", "no"], index=1)
    asthma = st.selectbox("Asthma Status", ["never_asthma", "current_asthma", "former_asthma"], index=0)
    could_not_afford_to_see_doctor = st.selectbox("Unable to See Doctor Due to Cost", ["yes", "no"], index=1)

with col2:
    health_care_provider = st.selectbox("Primary Healthcare Provider", ["yes_only_one", "more_than_one", "no"], index=0)
    stroke = st.selectbox("History of Stroke", ["yes", "no"], index=1, help="Stroke occurs when blood supply to the brain is interrupted")
    diabetes = st.selectbox("Diabetes Diagnosis", ["yes", "no", "no_prediabetes", "yes_during_pregnancy"], index=1)
    bmi = st.selectbox("Body Mass Index (BMI)", [
        "underweight_bmi_less_than_18_5", "normal_weight_bmi_18_5_to_24_9", "overweight_bmi_25_to_29_9",  
        "obese_bmi_30_or_more"
    ], index=1, help="Calculate your BMI at https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm")
    length_of_time_since_last_routine_checkup = st.selectbox("Last Routine Checkup", ["past_year", "past_2_years", "past_5_years", "5+_years_ago", "never"], index=0)

with col3:
    depressive_disorder = st.selectbox("Depressive Disorder Diagnosis", ["yes", "no"], index=1, help="Medical condition with persistent sadness and loss of interest")
    physical_health = st.selectbox("Physical Health (Past 30 Days)", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], index=0)
    mental_health = st.selectbox("Mental Health (Past 30 Days)", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], index=0)
    walking = st.selectbox("Difficulty Walking/Climbing Stairs", ["yes", "no"], index=1)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown("#### 🏃‍♂️ Lifestyle Factors")

col1, col2, col3 = st.columns(3)

with col1:
    smoking_status = st.selectbox("Smoking Status", ["never_smoked", "former_smoker", "current_smoker_some_days", "current_smoker_every_day"], index=0)
    sleep_category = st.selectbox("Sleep Duration (Typical Night)", [
        "very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours", "normal_sleep_6_to_8_hours",  
        "long_sleep_9_to_10_hours", "very_long_sleep_11_or_more_hours"], index=2)

with col2:
    drinks_category = st.selectbox("Weekly Alcohol Consumption", [
        "did_not_drink", "very_low_consumption_0.01_to_1_drinks", "low_consumption_1.01_to_5_drinks",  
        "moderate_consumption_5.01_to_10_drinks", "high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"], index=0)
    binge_drinking_status = st.selectbox("Binge Drinking (Past 30 Days)", ["yes", "no"], index=1, help="5+ drinks for men, 4+ drinks for women in ~2 hours")

with col3:
    exercise_status = st.selectbox("Exercise (Past 30 Days)", ["yes", "no"], index=0)

st.markdown('</div>', unsafe_allow_html=True)

# Collect input data
input_data = {
    'gender': gender,
    'race': race,
    'general_health': general_health,
    'health_care_provider': health_care_provider,
    'could_not_afford_to_see_doctor': could_not_afford_to_see_doctor,
    'length_of_time_since_last_routine_checkup': length_of_time_since_last_routine_checkup,
    'ever_diagnosed_with_heart_attack': heart_attack,
    'ever_diagnosed_with_a_stroke': stroke,
    'ever_told_you_had_a_depressive_disorder': depressive_disorder,
    'ever_told_you_have_kidney_disease': kidney_disease,
    'ever_told_you_had_diabetes': diabetes,
    'BMI': bmi,
    'difficulty_walking_or_climbing_stairs': walking,
    'physical_health_status': physical_health,
    'mental_health_status': mental_health,
    'asthma_Status': asthma,
    'smoking_status': smoking_status,
    'binge_drinking_status': binge_drinking_status,
    'exercise_status_in_past_30_Days': exercise_status,
    'age_category': age_category,
    'sleep_category': sleep_category,
    'drinks_category': drinks_category
}

def predict_heart_disease_risk(input_data, model, encoder):
    input_df = pd.DataFrame([input_data])
    input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
    prediction = model.predict_proba(input_encoded)[:, 1][0] * 100
    return prediction

st.markdown("---")

# Assessment Button
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
""", unsafe_allow_html=True)

if st.button('🚀 Get AI-Powered Risk Assessment', key='assessment_btn'):
    try:
        risk = predict_heart_disease_risk(input_data, model, encoder)
        
        # Determine risk level and styling
        if risk > 70:
            risk_class = "risk-high"
            risk_emoji = "🔴"
            risk_text = "Very High Risk"
        elif risk > 40:
            risk_class = "risk-high"
            risk_emoji = "🟠"
            risk_text = "High Risk"
        elif risk > 25:
            risk_class = "risk-moderate"
            risk_emoji = "🟡"
            risk_text = "Moderate Risk"
        else:
            risk_class = "risk-low"
            risk_emoji = "🟢"
            risk_text = "Low Risk"
        
        # Results Display
        st.markdown(f"""
        <div class="results-card">
            <h3 style="color: #06b6d4; text-align: center; margin-bottom: 1rem;">🤖 AI Assessment Results</h3>
            <div class="risk-score {risk_class}">
                {risk_emoji} {risk:.1f}% Risk
                <br><small style="font-size: 1rem; opacity: 0.8;">{risk_text}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP Analysis and Recommendations
        col1, col2 = st.columns([1, 1])
        
        with col2:
            # Generate SHAP values and feature importance
            input_df = pd.DataFrame([input_data])
            input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
            lgbm_model = model.estimators_[0].steps[-1][1]
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(input_encoded)
            feature_importances = np.abs(shap_values[1]).sum(axis=0)
            feature_importances /= feature_importances.sum()
            feature_importances *= 100
            feature_importance_df = pd.DataFrame({
                'Feature': input_encoded.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            # Create modern pie chart
            top_features = feature_importance_df.head(6)
            other_importance = feature_importance_df.iloc[6:]['Importance'].sum()
            
            if other_importance > 0:
                chart_data = pd.concat([top_features, pd.DataFrame({'Feature': ['Other Factors'], 'Importance': [other_importance]})], ignore_index=True)
            else:
                chart_data = top_features
            
            # Create a modern-looking pie chart with HoloMed AI colors
            fig = go.Figure(data=[go.Pie(
                labels=chart_data['Feature'], 
                values=chart_data['Importance'],
                hole=0.4,
                marker=dict(
                    colors=['#06b6d4', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#6b7280'],
                    line=dict(color='#0f172a', width=2)
                ),
                textfont=dict(color='white', size=12),
                hovertemplate='<b>%{label}</b><br>Contribution: %{value:.1f}%<extra></extra>'
            )])
            
            fig.update_layout(
                title=dict(
                    text="🔍 Risk Factor Analysis",
                    font=dict(color='#06b6d4', size=18, family='Inter'),
                    x=0.5
                ),
                paper_bgcolor='rgba(15, 23, 42, 0.9)',
                plot_bgcolor='rgba(15, 23, 42, 0.9)',
                font=dict(color='white', family='Inter'),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(color='#cbd5e1')
                ),
                margin=dict(t=80, b=20, l=20, r=120),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.9); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(6, 182, 212, 0.3); height: 400px; overflow-y: auto;">
                <h4 style="color: #06b6d4; margin-bottom: 1rem;">💡 AI-Powered Recommendations</h4>
            """, unsafe_allow_html=True)
            
            if risk > 25:
                # Generate specific recommendations based on risk factors
                recommendations = []
                
                # Get top contributing factors
                important_factors = feature_importance_df.head(5)
                
                for _, row in important_factors.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    
                    if feature == 'ever_diagnosed_with_heart_attack' and heart_attack == "yes":
                        st.markdown(f"""
                        <div class="recommendation">
                            <strong>🏥 Heart Attack History ({importance:.1f}% contribution)</strong><br>
                            Maintain regular cardiology visits and strict medication adherence. Monitor for new symptoms.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif feature == 'smoking_status' and smoking_status != "never_smoked":
                        st.markdown(f"""
                        <div class="recommendation">
                            <strong>🚭 Smoking ({importance:.1f}% contribution)</strong><br>
                            Quitting smoking is the single most effective way to reduce your cardiovascular risk. Seek professional help.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif feature == 'exercise_status_in_past_30_Days' and exercise_status == "no":
                        st.markdown(f"""
                        <div class="recommendation">
                            <strong>🏃‍♂️ Physical Activity ({importance:.1f}% contribution)</strong><br>
                            Start with 150 minutes of moderate exercise weekly. Even light walking significantly improves heart health.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif feature == 'BMI' and bmi in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]:
                        st.markdown(f"""
                        <div class="recommendation">
                            <strong>⚖️ Weight Management ({importance:.1f}% contribution)</strong><br>
                            Achieve healthy weight through balanced nutrition and regular exercise. Consult a healthcare provider for guidance.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif feature == 'ever_told_you_had_diabetes' and diabetes == "yes":
                        st.markdown(f"""
                        <div class="recommendation">
                            <strong>🩺 Diabetes Management ({importance:.1f}% contribution)</strong><br>
                            Maintain optimal blood sugar control through diet, exercise, and medication compliance.
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation">
                    <strong>🎉 Excellent Heart Health!</strong><br>
                    Your risk is low. Continue your healthy lifestyle habits to maintain optimal cardiovascular health.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ Medical Disclaimer</h4>
    <p>This AI-powered assessment is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers regarding your health concerns.</p>
</div>
""", unsafe_allow_html=True)

# Contact Section
with st.expander("💬 Connect with HoloMed AI"):
    st.markdown("""
    <div class="contact-form">
        <form action="https://formsubmit.co/your-email@domain.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your Name" required>
            <input type="email" name="email" placeholder="Your Email" required>
            <textarea name="message" placeholder="Ask us about AI in medicine..." rows="4" required></textarea>
            <button type="submit">Send Message</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="holomedai-footer">
    <h4>🌐 Follow HoloMed AI</h4>
    <div class="social-links">
        <a href="https://www.instagram.com/holomedai/" target="_blank">📱 Instagram</a>
        <a href="#" target="_blank">🔬 Research</a>
        <a href="#" target="_blank">📧 Contact</a>
        <a href="#" target="_blank">💡 About</a>
    </div>
    <p style="color: #64748b; margin-top: 1rem; font-size: 0.9rem;">
        © 2024 HoloMed AI. Advancing healthcare through artificial intelligence.
    </p>
    <p style="color: #64748b; font-size: 0.8rem;">
        Providing accessible education on the transformative impact of AI in Medicine
    </p>
</div>
""", unsafe_allow_html=True)