import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthFusion — Cardiovascular Risk Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main { background-color: #f6f9fc; }
      .hf-title { font-size: 2.2rem; font-weight: 700; color: #0f3460; margin-bottom: 0; }
      .hf-sub { color: #4a5568; margin-top: 0; }
      .hf-card {
          background: white; padding: 1.2rem 1.4rem; border-radius: 14px;
          box-shadow: 0 2px 10px rgba(15,52,96,0.07);
          border-left: 6px solid #0f3460; margin-bottom: 1rem;
      }
      .hf-card.high { border-left-color: #e63946; }
      .hf-card.low  { border-left-color: #2a9d8f; }
      .hf-metric-label { color: #6b7280; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
      .hf-metric-value { color: #0f3460; font-size: 1.6rem; font-weight: 700; }
      .hf-badge {
          display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
          font-weight: 600; font-size: 0.9rem;
      }
      .hf-badge.high { background: #fde2e4; color: #b8001f; }
      .hf-badge.low  { background: #d8f3dc; color: #1b6f4a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load model artifacts
# -----------------------------------------------------------------------------
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACT_DIR, "healthfusion_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "healthfusion_scaler.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))
    return model, scaler, list(feature_columns)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<p class="hf-title">❤️ HealthFusion</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hf-sub">AI-powered cardiovascular risk assessment combining clinical, '
    "lifestyle and environmental factors.</p>",
    unsafe_allow_html=True,
)
st.divider()

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model file. Place `healthfusion_model.pkl`, "
        "`healthfusion_scaler.pkl` and `feature_columns.pkl` next to `app.py`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Patient Inputs")

with st.sidebar.expander("👤 Demographics", expanded=True):
    age = st.number_input("Age (years)", 1, 120, 45)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (cm)", 80.0, 230.0, 170.0, step=0.5)
    weight = st.number_input("Weight (kg)", 20.0, 250.0, 75.0, step=0.5)

with st.sidebar.expander("🩺 Clinical", expanded=True):
    ap_hi = st.number_input("Systolic BP (ap_hi)", 60, 260, 120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", 30, 180, 80)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3],
                               format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
    gluc = st.selectbox("Glucose", [1, 2, 3],
                        format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])

with st.sidebar.expander("🏃 Lifestyle", expanded=False):
    smoke = st.radio("Smoking", ["No", "Yes"], horizontal=True)
    alco = st.radio("Alcohol Consumption", ["No", "Yes"], horizontal=True)
    active = st.radio("Physical Activity", ["No", "Yes"], horizontal=True, index=1)

with st.sidebar.expander("🌍 Environment", expanded=False):
    temperature = st.number_input("Temperature (°C)", -30.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 55.0)
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 35.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 50.0)
    no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 20.0)
    so2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 10.0)
    co = st.number_input("CO (mg/m³)", 0.0, 50.0, 1.0)
    proximity = st.slider("Proximity to Industrial Areas (km)", 0.0, 50.0, 10.0)
    pop_density = st.number_input("Population Density (people/km²)", 0.0, 100000.0, 3000.0)
    air_quality = st.selectbox(
        "Air Quality",
        [1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Good", 2: "Moderate", 3: "Unhealthy (Sensitive)",
                               4: "Unhealthy", 5: "Hazardous"}[x],
        index=1,
    )

predict_clicked = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")

# -----------------------------------------------------------------------------
# Derived features
# -----------------------------------------------------------------------------
bmi = weight / ((height / 100.0) ** 2)
health_stress_score = cholesterol + gluc + ap_hi / 100.0
heat_risk = temperature + humidity
pollution_risk = pm25 + pm10 + no2

# Mapping dictionary — keys cover common feature names
raw_features = {
    "age": age,
    "gender": 1 if gender == "Female" else 2,  # cardio dataset convention
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": 1 if smoke == "Yes" else 0,
    "alco": 1 if alco == "Yes" else 0,
    "active": 1 if active == "Yes" else 0,
    "Temperature": temperature,
    "Humidity": humidity,
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "SO2": so2,
    "CO": co,
    "Proximity_to_Industrial_Areas": proximity,
    "Population_Density": pop_density,
    "Air_Quality": air_quality,
    "BMI": bmi,
    "Health_Stress_Score": health_stress_score,
    "Heat_Risk": heat_risk,
    "Pollution_Risk": pollution_risk,
}

# -----------------------------------------------------------------------------
# Top metrics
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("BMI", f"{bmi:.1f}")
c2.metric("Health Stress Score", f"{health_stress_score:.2f}")
c3.metric("Heat Risk", f"{heat_risk:.1f}")
c4.metric("Pollution Risk", f"{pollution_risk:.1f}")

st.write("")

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
def build_input_row(columns):
    row = {}
    for col in columns:
        if col in raw_features:
            row[col] = raw_features[col]
        else:
            # case-insensitive fallback
            match = next((k for k in raw_features if k.lower() == col.lower()), None)
            row[col] = raw_features[match] if match else 0
    return pd.DataFrame([row], columns=columns)

if predict_clicked:
    X = build_input_row(feature_columns)
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Scaler error: {e}")
        st.stop()

    proba = float(model.predict_proba(X_scaled)[0][1])
    pred = int(proba >= 0.5)
    label = "High Risk" if pred == 1 else "Low Risk"
    badge_class = "high" if pred == 1 else "low"

    st.markdown(
        f'<div class="hf-card {badge_class}">'
        f'<span class="hf-metric-label">Prediction</span><br/>'
        f'<span class="hf-badge {badge_class}">{label}</span>'
        f'&nbsp;&nbsp;<span class="hf-metric-value">{proba*100:.1f}%</span>'
        f' <span style="color:#6b7280;font-size:0.9rem;">probability of cardiovascular risk</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(proba, 0.0), 1.0))

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="hf-card">', unsafe_allow_html=True)
        st.subheader("🩺 Health Summary")
        st.write(f"**Age:** {age}  |  **Gender:** {gender}")
        st.write(f"**BMI:** {bmi:.1f} ({'Underweight' if bmi<18.5 else 'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'})")
        st.write(f"**Blood Pressure:** {ap_hi}/{ap_lo} mmHg")
        st.write(f"**Cholesterol level:** {cholesterol}  |  **Glucose level:** {gluc}")
        st.write(f"**Lifestyle:** Smoke={smoke}, Alcohol={alco}, Active={active}")
        st.write(f"**Health Stress Score:** {health_stress_score:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="hf-card">', unsafe_allow_html=True)
        st.subheader("🌍 Environmental Risk Summary")
        st.write(f"**Temperature:** {temperature} °C  |  **Humidity:** {humidity}%")
        st.write(f"**Heat Risk Index:** {heat_risk:.1f}")
        st.write(f"**PM2.5 / PM10:** {pm25} / {pm10} µg/m³")
        st.write(f"**NO2 / SO2 / CO:** {no2} / {so2} / {co}")
        st.write(f"**Air Quality:** {air_quality}  |  **Pollution Risk:** {pollution_risk:.1f}")
        st.write(f"**Industrial proximity:** {proximity} km  |  **Pop. density:** {pop_density:.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔬 Model input (debug)"):
        st.dataframe(X.T.rename(columns={0: "value"}))
else:
    st.info("Fill out patient inputs in the sidebar and click **Predict Risk**.")

st.divider()
st.caption("HealthFusion is a decision-support tool and is not a substitute for professional medical advice.")