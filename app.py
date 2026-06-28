import streamlit as st
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------
# Page Settings
# -------------------------------

st.set_page_config(
    page_title="Smart Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

st.sidebar.title("📱 Project Information")

st.sidebar.success("Machine Learning Lab Project")

st.sidebar.write("""
### Model Used
Gradient Boosting Classifier

### Dataset
5000 Smartphone Records

### Classes
• Healthy
• Moderate
• Poor

### Technologies
• Python
• Pandas
• Scikit-Learn
• Streamlit
""")

st.title("🔋 Smart Battery Health Prediction & Maintenance Advisor")

st.write("""
Predict smartphone battery health using Machine Learning.
""")
st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", "5,000")
col2.metric("ML Model", "Gradient Boosting")
col3.metric("Prediction Classes", "3")
# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("battery_health_processed.csv")

# Features and Target
X = df.drop(columns=["battery_health", "battery_health_score"])
y = df["battery_health"]

# Train Model
model = GradientBoostingClassifier(random_state=42)
model.fit(X, y)

st.success("✅ Model trained successfully!")
st.markdown("---")
st.header("📱 Enter Smartphone Information")

col1, col2 = st.columns(2)

with col1:

    device_age = st.number_input(
        "Device Age (Months)",
        min_value=1,
        max_value=120,
        value=24
    )

    battery_capacity = st.number_input(
        "Battery Capacity (mAh)",
        min_value=2000,
        max_value=7000,
        value=5000
    )

    screen_time = st.slider(
        "Average Screen Time (Hours/Day)",
        1.0, 15.0, 6.0
    )

    charging_cycles = st.slider(
        "Charging Cycles per Week",
        1.0, 30.0, 10.0
    )

    battery_temp = st.slider(
        "Battery Temperature (°C)",
        20.0, 60.0, 35.0
    )

    fast_charge = st.slider(
        "Fast Charging Usage (%)",
        0.0, 100.0, 40.0
    )

    overnight = st.slider(
        "Overnight Charging / Week",
        0, 7, 2
    )

with col2:

    gaming = st.slider(
        "Gaming Hours / Week",
        0.0, 40.0, 8.0
    )

    video = st.slider(
        "Video Streaming Hours / Week",
        0.0, 40.0, 10.0
    )

    background = st.selectbox(
        "Background App Usage",
        ["Low", "Medium", "High"]
    )

    signal = st.selectbox(
        "Signal Strength",
        ["Poor", "Moderate", "Good"]
    )

    charging_habit = st.slider(
        "Charging Habit Score",
        0, 100, 70
    )

    usage_score = st.slider(
        "Usage Intensity Score",
        0.0, 100.0, 50.0
    )

    thermal = st.slider(
        "Thermal Stress Index",
        0.0, 100.0, 40.0
    )

     
background_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

signal_mapping = {
    "Poor": 1,
    "Moderate": 2,
    "Good": 3
}

if st.button("🔋 Predict Battery Health", use_container_width=True):

    input_data = pd.DataFrame([{

        "device_age_months": device_age,
        "battery_capacity_mah": battery_capacity,
        "avg_screen_on_hours_per_day": screen_time,
        "avg_charging_cycles_per_week": charging_cycles,
        "avg_battery_temp_celsius": battery_temp,
        "fast_charging_usage_percent": fast_charge,
        "overnight_charging_freq_per_week": overnight,
        "gaming_hours_per_week": gaming,
        "video_streaming_hours_per_week": video,
        "background_app_usage_level": background_mapping[background],
        "signal_strength_avg": signal_mapping[signal],
        "charging_habit_score": charging_habit,
        "usage_intensity_score": usage_score,
        "thermal_stress_index": thermal

    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max() * 100
    battery_score = max(
    0,
    min(
        100,
        100
        - device_age * 0.4
        - battery_temp * 0.6
        - fast_charge * 0.15
        - charging_cycles * 0.5
        - thermal * 0.3
        + charging_habit * 0.2,
    ),
) 
with st.expander("📋 Input Summary"):
    st.dataframe(input_data)


st.markdown("---")

if prediction == "Healthy":
        st.success(f"🟢 Battery Health: **{prediction}**")

elif prediction == "Moderate":
        st.warning(f"🟡 Battery Health: **{prediction}**")

else:
        st.error(f"🔴 Battery Health: **{prediction}**")

st.info(f"🎯 Prediction Confidence: {probability:.2f}%")
    

st.subheader("🔧 Maintenance Advice")

if prediction == "Healthy":

        st.success("""
✅ Your battery is in excellent condition.

• Continue avoiding extreme temperatures.
• Avoid keeping your phone plugged in overnight.
• Keep fast charging usage moderate.
• Maintain healthy charging habits.
""")
        st.balloons()

elif prediction == "Moderate":

        st.warning("""
⚠ Your battery is showing signs of wear.

• Reduce fast charging frequency.
• Avoid excessive gaming while charging.
• Lower battery temperature whenever possible.
• Charge between 20% and 80%.
""")

else:

        st.error("""
🚨 Battery health is poor.

• Consider replacing the battery.
• Avoid overheating.
• Reduce heavy gaming sessions.
• Minimize overnight charging.
• Backup important data regularly.
""")
        
st.markdown("---")

st.info(
    "⚠️ This prediction is generated using a Machine Learning model trained on a simulated smartphone battery dataset and is intended for educational purposes."
)

st.subheader("🔋 Estimated Battery Health Score")

st.progress(int(battery_score))

st.write(f"### {battery_score:.1f} / 100")

st.caption(
    "Developed as Final Machine Learning Lab Project | Smart Battery Health Prediction & Maintenance Advisor"
)