import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from sklearn.ensemble import VotingRegressor, StackingRegressor

warnings.filterwarnings('ignore')

# Function to apply feature engineering (crucial for consistency)
def engineer_features(df):
    """Apply consistent feature engineering to both training and prediction data"""
    # Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    
    # Basic feature engineering
    processed_df['Soil_Texture_Index'] = processed_df['NIR_Spectroscopy_900nm'] / processed_df['NIR_Spectroscopy_2500nm']
    processed_df['NPK_Total'] = processed_df['Nutrient_Nitrogen_mg_kg'] + processed_df['Nutrient_Phosphorus_mg_kg'] + processed_df['Nutrient_Potassium_mg_kg']
    processed_df['NPK_Ratio'] = processed_df['NPK_Total'] / 3
    processed_df['pH_Balance'] = np.abs(processed_df['pH_Level'] - 6.5)  # Deviation from neutral pH
    processed_df['NIR_Ratio'] = processed_df['NIR_Spectroscopy_900nm'] / processed_df['NIR_Spectroscopy_2500nm']
    processed_df['Visible_Ratio'] = processed_df['Visible_Light_400nm'] / processed_df['Visible_Light_700nm']
    processed_df['Temp_Moisture_Interaction'] = processed_df['Temperature_C'] * processed_df['Moisture_Content_%']
    processed_df['EC_pH_Interaction'] = processed_df['Electrical_Conductivity_dS_m'] * processed_df['pH_Level']
    processed_df['Nitrogen_Phosphorus_Ratio'] = processed_df['Nutrient_Nitrogen_mg_kg'] / (processed_df['Nutrient_Phosphorus_mg_kg'] + 0.001)
    processed_df['Organic_pH_Interaction'] = processed_df['Organic_Matter_%'] * processed_df['pH_Level']
    
    # Time features
    processed_df['Time_Sine'] = np.sin(2 * np.pi * processed_df['Time_of_Measurement'] / 24)
    processed_df['Time_Cosine'] = np.cos(2 * np.pi * processed_df['Time_of_Measurement'] / 24)
    processed_df['Time_Category'] = pd.cut(processed_df['Time_of_Measurement'] % 24,
                                bins=[0, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    # Location-based features
    processed_df['Location_Cluster'] = processed_df.apply(
        lambda row: f"{round(row['GPS_Latitude'], 1)}_{round(row['GPS_Longitude'], 1)}", axis=1)
    
    return processed_df

def define_features_targets(df):
    """Define the feature set and target variables consistently"""
    # Define features by dropping target columns
    features = df.drop(columns=[
        "Fertility_Score", "Soil_Type", "Water_Retention_Capacity",
        "Lime_Requirement", "Soil_Erosion_Risk",
        "Nutrient_Nitrogen_mg_kg", "Nutrient_Phosphorus_mg_kg",
        "Nutrient_Potassium_mg_kg", "Organic_Matter_%"
    ], errors='ignore')
    
    # Define regression targets
    targets_regression = df[[
        "Fertility_Score", "Nutrient_Nitrogen_mg_kg", "Nutrient_Phosphorus_mg_kg",
        "Nutrient_Potassium_mg_kg", "Organic_Matter_%", "Water_Retention_Capacity",
        "Lime_Requirement", "Soil_Erosion_Risk"
    ]].dropna()
    
    # Define classification target
    targets_classification = df['Soil_Type'].dropna()
    
    # Match features to target rows
    features_reg = features.loc[targets_regression.index]
    features_cls = features.loc[targets_classification.index]
    
    return features_reg, targets_regression, features_cls, targets_classification

def create_preprocessor(features_df):
    """Create consistent preprocessing pipeline"""
    # Identify feature types
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create pipelines
    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer with remainder='passthrough' to handle any extra columns
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')  # Drop columns not specified - important for consistency
    
    return preprocessor, numerical_features, categorical_features

def train_and_save_models(file_path):
    """Complete pipeline to train and save models with consistent preprocessing"""
    # Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Engineer features
    print("Engineering features...")
    df_processed = engineer_features(df)
    
    # Define features and targets
    print("Defining features and targets...")
    features_reg, targets_regression, features_cls, targets_classification = define_features_targets(df_processed)
    
    # Save feature column structure
    feature_info = {
        'columns': features_reg.columns.tolist(),
        'dtypes': {col: str(features_reg[col].dtype) for col in features_reg.columns}
    }
    joblib.dump(feature_info, "feature_info.pkl")
    
    # Split data
    print("Splitting data...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        features_reg, targets_regression, test_size=0.2, random_state=42)
    
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        features_cls, targets_classification, test_size=0.2, random_state=42)
    
    # Create preprocessor
    print("Creating preprocessor...")
    preprocessor, numerical_features, categorical_features = create_preprocessor(features_reg)
    
    # Train models
    print("Training models...")
    models = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)),
        "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        "LightGBM": MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    }
    
    # Evaluate and save best model
    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        
        # Train model
        print(f"Training {name}...")
        pipeline.fit(X_train_reg, y_train_reg)
        
        # Evaluate
        y_pred = pipeline.predict(X_test_reg)
        r2 = r2_score(y_test_reg, y_pred, multioutput='uniform_average')
        results[name] = {'model': pipeline, 'r2_score': r2}
        print(f"{name} - R²: {r2:.4f}")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2_score'])
    best_model = results[best_model_name]['model']
    print(f"Best model: {best_model_name} with R² = {results[best_model_name]['r2_score']:.4f}")
    
    # Save best regression model
    print("Saving regression model...")
    joblib.dump(best_model, "final_combined_regression_model.pkl")
    
    # Create, train and save classification model
    print("Training classification model...")
    cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cls_pipeline = Pipeline([('preprocessor', preprocessor), ('model', cls_model)])
    cls_pipeline.fit(X_train_cls, y_train_cls)
    
    # Evaluate classification model
    cls_pred = cls_pipeline.predict(X_test_cls)
    cls_accuracy = accuracy_score(y_test_cls, cls_pred)
    print(f"Classification model accuracy: {cls_accuracy:.4f}")
    
    # Save classification model
    print("Saving classification model...")
    joblib.dump(cls_pipeline, "final_combined_classification_model.pkl")
    
    return best_model, cls_pipeline, preprocessor, feature_info

def prepare_input_for_prediction(input_data, feature_info):
    """Prepare input data for prediction ensuring column consistency"""
    # Engineer features
    processed_input = engineer_features(input_data)
    
    # Ensure column order and presence matches training data
    expected_columns = feature_info['columns']
    
    # Create a DataFrame with the expected columns, filling missing ones with NaN
    aligned_input = pd.DataFrame(columns=expected_columns)
    for col in expected_columns:
        if col in processed_input.columns:
            aligned_input[col] = processed_input[col]
        else:
            aligned_input[col] = np.nan
    
    return aligned_input

def make_predictions(input_data):
    """Make predictions ensuring consistent preprocessing"""
    # Load models and feature info
    regression_model = joblib.load("final_combined_regression_model.pkl")
    classification_model = joblib.load("final_combined_classification_model.pkl")
    feature_info = joblib.load("feature_info.pkl")
    
    # Prepare input data with consistent columns
    prepared_input = prepare_input_for_prediction(input_data, feature_info)
    
    # Make predictions
    regression_predictions = regression_model.predict(prepared_input)
    classification_predictions = classification_model.predict(prepared_input)
    
    # Create results DataFrame
    reg_columns = ["Fertility_Score", "Nutrient_Nitrogen_mg_kg", "Nutrient_Phosphorus_mg_kg",
                  "Nutrient_Potassium_mg_kg", "Organic_Matter_%", "Water_Retention_Capacity",
                  "Lime_Requirement", "Soil_Erosion_Risk"]
    
    results_df = pd.DataFrame(regression_predictions, columns=reg_columns)
    results_df["Soil_Type"] = classification_predictions
    
    return results_df

if __name__ == "__main__":
    # Set the file path to your data
    file_path = "final_combined_soil_dataset.csv"
    
    # Train and save models
    train_and_save_models(file_path)
