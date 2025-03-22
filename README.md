project link https://ai-soil-analysis-hilm9ff2jw9te5ttxntpfr.streamlit.app/

# Soil Analysis AI - Streamlit Application

This application uses machine learning to analyze soil properties and provide insights for agricultural decisions.

## Problem Solved

The original code had an issue where the preprocessing columns during training didn't match those during prediction, causing errors. The refactored code provides consistent feature engineering and preprocessing across both training and inference stages.

## Project Structure

```
soil_analysis_app/
│
├── ai.py                             # Core ML code with the fixed preprocessing
├── app.py                            # Streamlit application
├── train.py                          # Script to train and save models
├── requirements.txt                  # Dependencies
├── final_combined_soil_dataset.csv   # Training data
├── final_combined_regression_model.pkl  # Saved regression model (after training)
├── final_combined_classification_model.pkl  # Saved classification model (after training)
└── feature_info.pkl                  # Saved feature column information (after training)
```

## How to Run

1. Make sure you have Python 3.8+ installed

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv soil_venv
   source soil_venv/bin/activate  # On Windows: soil_venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Train the models (only needed once):
   ```
   python train.py
   ```

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

6. Open your browser and go to the URL shown in the terminal (typically http://localhost:8501)

## Key Features

- **Soil Property Prediction**: Estimates fertility, nutrients, and more
- **Soil Type Classification**: Identifies soil types based on measurements
- **Data Visualization**: Shows soil data through informative charts
- **Batch Processing**: Upload multiple soil samples for analysis at once
- **Single Sample Analysis**: Enter individual soil measurements manually

## Technical Details

### Fixed Issues

The main issue fixed in this version is the column mismatch during preprocessing. The solution includes:

1. Consistent feature engineering via the `engineer_features()` function
2. Saving feature column information during training
3. Ensuring prediction inputs match the expected column structure
4. Using the same preprocessing pipeline for both training and prediction

### ML Pipeline

The application uses a scikit-learn pipeline with:

1. Feature engineering for derived soil properties
2. KNN imputation for missing values
3. Robust scaling for numerical features
4. One-hot encoding for categorical features
5. Multi-output regression for numerical predictions
6. Random forest classification for soil type prediction

## Data Requirements

The minimum required soil measurements are:

- NIR Spectroscopy (900nm, 2500nm)
- Visible Light (400nm, 700nm)
- Temperature (°C)
- Moisture Content (%)
- pH Level
- Electrical Conductivity (dS/m)
- GPS coordinates (Latitude, Longitude)
- Time of Measurement
- Nutrient levels (Nitrogen, Phosphorus, Potassium)
- Organic Matter (%)

An example template is available for download within the app.
