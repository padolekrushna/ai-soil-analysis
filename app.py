import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load models
classification_model = joblib.load('model/final_combined_classification_model.pkl')
regression_model = joblib.load('model/final_combined_regression_model.pkl')

# Load dataset
dataset = pd.read_csv('data/final_combined_soil_dataset.csv')

# Streamlit Interface
st.title('Soil Analysis Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    # Define all 18 features, including default values
    NIR_Spectroscopy_900nm = st.sidebar.number_input('NIR_Spectroscopy_900nm', value=0.0)
    NIR_Spectroscopy_2500nm = st.sidebar.number_input('NIR_Spectroscopy_2500nm', value=0.0)
    Nutrient_Nitrogen_mg_kg = st.sidebar.number_input('Nutrient_Nitrogen_mg_kg', value=0.0)
    Nutrient_Phosphorus_mg_kg = st.sidebar.number_input('Nutrient_Phosphorus_mg_kg', value=0.0)
    Nutrient_Potassium_mg_kg = st.sidebar.number_input('Nutrient_Potassium_mg_kg', value=0.0)
    pH_Level = st.sidebar.number_input('pH_Level', value=7.0)
    Visible_Light_400nm = st.sidebar.number_input('Visible_Light_400nm', value=0.0)
    Visible_Light_700nm = st.sidebar.number_input('Visible_Light_700nm', value=0.0)
    Temperature_C = st.sidebar.number_input('Temperature_C', value=25.0)
    Moisture_Content_ = st.sidebar.number_input('Moisture_Content_%', value=0.0)
    Electrical_Conductivity_dS_m = st.sidebar.number_input('Electrical_Conductivity_dS_m', value=0.0)
    Organic_Matter_ = st.sidebar.number_input('Organic_Matter_%', value=0.0)
    GPS_Latitude = st.sidebar.number_input('GPS_Latitude', value=0.0)
    GPS_Longitude = st.sidebar.number_input('GPS_Longitude', value=0.0)
    Time_of_Measurement = st.sidebar.number_input('Time_of_Measurement', value=0.0)

    # Placeholder for any missing features
    feature_17 = 0.0  # Add missing feature here
    feature_18 = 0.0  # Add missing feature here
    feature_19 = 0.0

    features = {
        'NIR_Spectroscopy_900nm': NIR_Spectroscopy_900nm,
        'NIR_Spectroscopy_2500nm': NIR_Spectroscopy_2500nm,
        'Nutrient_Nitrogen_mg_kg': Nutrient_Nitrogen_mg_kg,
        'Nutrient_Phosphorus_mg_kg': Nutrient_Phosphorus_mg_kg,
        'Nutrient_Potassium_mg_kg': Nutrient_Potassium_mg_kg,
        'pH_Level': pH_Level,
        'Visible_Light_400nm': Visible_Light_400nm,
        'Visible_Light_700nm': Visible_Light_700nm,
        'Temperature_C': Temperature_C,
        'Moisture_Content_%': Moisture_Content_,
        'Electrical_Conductivity_dS_m': Electrical_Conductivity_dS_m,
        'Organic_Matter_%': Organic_Matter_,
        'GPS_Latitude': GPS_Latitude,
        'GPS_Longitude': GPS_Longitude,
        'Time_of_Measurement': Time_of_Measurement,
        'Feature_17': feature_17,  # Add missing feature
        'Feature_18': feature_18,   # Add missing feature
        'Feature_19': feature_19
    }

    return pd.DataFrame(features, index=[0])

# User input features
input_data = user_input_features()

# Check if input_data is a pandas DataFrame
if not isinstance(input_data, pd.DataFrame):
    input_data = pd.DataFrame(input_data)

# ColumnTransformer setup with all 18 features
column_transformer = ColumnTransformer(
    transformers=[
        ('imputer', SimpleImputer(strategy='mean'), [
            'NIR_Spectroscopy_900nm', 'NIR_Spectroscopy_2500nm', 'Nutrient_Nitrogen_mg_kg', 
            'Nutrient_Phosphorus_mg_kg', 'Nutrient_Potassium_mg_kg', 'pH_Level', 
            'Visible_Light_400nm', 'Visible_Light_700nm', 'Temperature_C', 
            'Moisture_Content_%', 'Electrical_Conductivity_dS_m', 'Organic_Matter_%', 
            'GPS_Latitude', 'GPS_Longitude', 'Time_of_Measurement', 'Feature_17', 'Feature_18', 'Feature_19'
        ]),
    ])

# Apply the transformer to the input data
input_data_transformed = column_transformer.fit_transform(input_data)

# Preprocess input data (standard scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_data_transformed)

# Model predictions
classification_pred = classification_model.predict(scaled_data)
regression_pred = regression_model.predict(scaled_data)

# Display predictions
st.subheader('Classification Prediction (e.g., Soil Fertility Level)')
st.write(classification_pred)

st.subheader('Regression Predictions (e.g., Nutrient Levels, Organic Matter, etc.)')
st.write(f'Nitrogen: {regression_pred[0][0]} mg/kg')
st.write(f'Phosphorus: {regression_pred[0][1]} mg/kg')
st.write(f'Potassium: {regression_pred[0][2]} mg/kg')
st.write(f'Organic Matter: {regression_pred[0][3]}%')
st.write(f'Water Retention Capacity: {regression_pred[0][4]}')
st.write(f'Lime Requirement: {regression_pred[0][5]}')
st.write(f'Soil Erosion Risk: {regression_pred[0][6]}')
