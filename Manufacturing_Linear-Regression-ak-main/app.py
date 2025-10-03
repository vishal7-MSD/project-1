import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load Model and Feature Order ---
model_path = 'linear_regression_model_all_features.pkl'
features_path = 'trained_features.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        trained_features = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("Manufacturing Parts Per Hour Predictor")
st.write("Predict number of parts produced per hour based on manufacturing parameters.")

# --- User Input ---
def user_input_features():
    st.sidebar.header("Input Parameters")

    # Base numerical inputs
    injection_temp = st.sidebar.slider('Injection Temperature', 150.0, 300.0, 220.0)
    injection_pressure = st.sidebar.slider('Injection Pressure', 100.0, 200.0, 130.0)
    cycle_time = st.sidebar.slider('Cycle Time', 15.0, 60.0, 30.0)
    cooling_time = st.sidebar.slider('Cooling Time', 5.0, 30.0, 15.0)
    material_viscosity = st.sidebar.slider('Material Viscosity', 200.0, 500.0, 350.0)
    ambient_temp = st.sidebar.slider('Ambient Temperature', 20.0, 35.0, 25.0)
    machine_age = st.sidebar.slider('Machine Age', 0.0, 20.0, 5.0)
    operator_exp = st.sidebar.slider('Operator Experience', 0.0, 25.0, 10.0)
    maintenance_hours = st.sidebar.slider('Maintenance Hours', 0.0, 100.0, 50.0)

    # Categoricals
    shift = st.sidebar.radio("Shift", ["Day", "Evening", "Night"])
    machine_type = st.sidebar.radio("Machine Type", ["Type_A", "Type_B", "Type_C"])
    material_grade = st.sidebar.radio("Material Grade", ["Economy", "Standard", "Premium"])
    day_of_week = st.sidebar.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    # --- Build DataFrame with same columns/order as training ---
    input_df = pd.DataFrame(np.zeros((1, len(trained_features))), columns=trained_features)

    # Fill numerical
    input_df['Injection_Temperature'] = injection_temp
    input_df['Injection_Pressure'] = injection_pressure
    input_df['Cycle_Time'] = cycle_time
    input_df['Cooling_Time'] = cooling_time
    input_df['Material_Viscosity'] = material_viscosity
    input_df['Ambient_Temperature'] = ambient_temp
    input_df['Machine_Age'] = machine_age
    input_df['Operator_Experience'] = operator_exp
    input_df['Maintenance_Hours'] = maintenance_hours

    # Engineered features (must match training)
    input_df['Temperature_Pressure_Ratio'] = injection_temp / injection_pressure
    input_df['Total_Cycle_Time'] = cycle_time + cooling_time
    input_df['Efficiency_Score'] = (injection_temp / injection_pressure) / cycle_time
    input_df['Machine_Utilization'] = input_df['Total_Cycle_Time'] / (input_df['Total_Cycle_Time'] + 10)

    # One-hot encoding
    if shift == "Evening":
        col = "Shift_Evening"
        if col in input_df.columns: input_df[col] = 1
    elif shift == "Night":
        col = "Shift_Night"
        if col in input_df.columns: input_df[col] = 1

    if machine_type == "Type_B":
        col = "Machine_Type_Type_B"
        if col in input_df.columns: input_df[col] = 1
    elif machine_type == "Type_C":
        col = "Machine_Type_Type_C"
        if col in input_df.columns: input_df[col] = 1

    if material_grade == "Standard":
        col = "Material_Grade_Standard"
        if col in input_df.columns: input_df[col] = 1
    elif material_grade == "Premium":
        col = "Material_Grade_Premium"
        if col in input_df.columns: input_df[col] = 1

    if day_of_week != "Friday":
        col = f"Day_of_Week_{day_of_week}"
        if col in input_df.columns: input_df[col] = 1

    return input_df

# Collect user input
df_input = user_input_features()

# Show input
st.subheader("User Input Parameters")
st.dataframe(df_input)

# --- Prediction ---
try:
    prediction = model.predict(df_input)
    st.subheader("Predicted Parts Per Hour")
    st.write(f"**{prediction[0]:.2f} parts per hour** 🚀")
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.warning("Ensure trained_features.pkl and input feature order are correct.")
