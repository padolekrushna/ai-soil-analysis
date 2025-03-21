import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Soil Analysis Predictor", 
    page_icon="🌱", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
.highlight {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Models with Detailed Error Handling
@st.cache_resource
def load_models():
    try:
        regression_model = joblib.load('final_combined_regression_model.pkl')
        classification_model = joblib.load('final_combined_classification_model.pkl')
        
        # Verify model types and attributes
        if not hasattr(regression_model, 'predict'):
            raise AttributeError("Regression model lacks prediction method")
        if not hasattr(classification_model, 'predict'):
            raise AttributeError("Classification model lacks prediction method")
        
        return regression_model, classification_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        logger.error(f"Model loading error: {e}")
        return None, None

# Prediction Function with Robust Error Handling
def predict_soil_characteristics(input_data, regression_model, classification_model):
    try:
        # Verify input data columns
        expected_columns = [
            'NIR_Spectroscopy_900nm', 'NIR_Spectroscopy_2500nm', 
            'Visible_Light_400nm', 'Visible_Light_700nm', 
            'Temperature_C', 'Moisture_Content_%', 'pH_Level', 
            'Electrical_Conductivity_dS_m', 'GPS_Latitude', 
            'GPS_Longitude', 'Time_of_Measurement'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in expected_columns if col not in input_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Ensure only expected columns are used
        input_data = input_data[expected_columns]
        
        # Regression Prediction
        regression_predictions = regression_model.predict(input_data)[0]
        target_columns = [
            'Fertility_Score', 
            'Nutrient_Nitrogen_mg_kg', 
            'Nutrient_Phosphorus_mg_kg',
            'Nutrient_Potassium_mg_kg', 
            'Organic_Matter_%', 
            'Water_Retention_Capacity',
            'Lime_Requirement', 
            'Soil_Erosion_Risk'
        ]
        regression_results = dict(zip(target_columns, regression_predictions))
        
        # Classification Prediction
        soil_type = classification_model.predict(input_data)[0]
        
        return regression_results, soil_type
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# Main Streamlit App
def main():
    st.title("🌍 Soil Analysis Predictor")
    st.markdown("### Advanced Machine Learning Model for Soil Characterization")
    
    # Load Models
    regression_model, classification_model = load_models()
    
    if regression_model is None or classification_model is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Sidebar for Input
    st.sidebar.header("Soil Parameters Input")
    
    # Input Features with Default Values and Ranges
    input_features = {
        'NIR_Spectroscopy_900nm': st.sidebar.number_input('NIR Spectroscopy 900nm', min_value=0.0, max_value=10.0, value=0.5, step=0.01),
        'NIR_Spectroscopy_2500nm': st.sidebar.number_input('NIR Spectroscopy 2500nm', min_value=0.0, max_value=10.0, value=0.3, step=0.01),
        'Visible_Light_400nm': st.sidebar.number_input('Visible Light 400nm', min_value=0.0, max_value=10.0, value=0.2, step=0.01),
        'Visible_Light_700nm': st.sidebar.number_input('Visible Light 700nm', min_value=0.0, max_value=10.0, value=0.3, step=0.01),
        'Temperature_C': st.sidebar.number_input('Temperature (°C)', min_value=-50.0, max_value=100.0, value=25.0, step=0.1),
        'Moisture_Content_%': st.sidebar.number_input('Moisture Content (%)', min_value=0.0, max_value=100.0, value=30.0, step=0.1),
        'pH_Level': st.sidebar.number_input('pH Level', min_value=0.0, max_value=14.0, value=7.0, step=0.1),
        'Electrical_Conductivity_dS_m': st.sidebar.number_input('Electrical Conductivity (dS/m)', min_value=0.0, max_value=10.0, value=0.5, step=0.01),
        'GPS_Latitude': st.sidebar.number_input('GPS Latitude', min_value=-90.0, max_value=90.0, value=0.0, step=0.001),
        'GPS_Longitude': st.sidebar.number_input('GPS Longitude', min_value=-180.0, max_value=180.0, value=0.0, step=0.001),
        'Time_of_Measurement': st.sidebar.number_input('Time of Measurement', min_value=0, max_value=24, value=12, step=1)
    }
    
    # Create prediction button
    if st.sidebar.button('Predict Soil Characteristics'):
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_features])
            
            # Get Predictions
            regression_results, soil_type = predict_soil_characteristics(
                input_df, regression_model, classification_model
            )
            
            # Display Results
            st.subheader("🔬 Prediction Results")
            
            # Regression Results
            st.markdown("#### Soil Regression Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Fertility Score", f"{regression_results['Fertility_Score']:.2f}")
                st.metric("Nutrient Nitrogen", f"{regression_results['Nutrient_Nitrogen_mg_kg']:.2f} mg/kg")
                st.metric("Nutrient Phosphorus", f"{regression_results['Nutrient_Phosphorus_mg_kg']:.2f} mg/kg")
                st.metric("Water Retention", f"{regression_results['Water_Retention_Capacity']:.2f}")
            
            with col2:
                st.metric("Nutrient Potassium", f"{regression_results['Nutrient_Potassium_mg_kg']:.2f} mg/kg")
                st.metric("Organic Matter", f"{regression_results['Organic_Matter_%']:.2f}%")
                st.metric("Lime Requirement", f"{regression_results['Lime_Requirement']:.2f}")
                st.metric("Soil Erosion Risk", f"{regression_results['Soil_Erosion_Risk']:.2f}")
            
            # Soil Type Classification
            st.markdown("#### Soil Type Classification")
            st.success(f"Predicted Soil Type: {soil_type}")
            
        except ValueError as ve:
            st.error(f"Input Error: {ve}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            logger.error(f"Unexpected error in prediction: {e}")

    # About Section
    st.sidebar.markdown("### About the Model")
    st.sidebar.info(
        "This AI model predicts soil characteristics based on various spectroscopic, "
        "environmental, and geographical features. It provides insights into soil fertility, "
        "nutrient content, and potential agricultural suitability."
    )

if __name__ == "__main__":
    main()
