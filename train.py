# Script to train models using the refactored code
from ai import train_and_save_models
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("Starting model training...")
    
    # Set file path to the dataset
    file_path = "final_combined_soil_dataset.csv"
    
    # Train and save models
    models = train_and_save_models(file_path)
    
    print("\nTraining completed successfully!")
    print("The following files have been created:")
    print("- final_combined_regression_model.pkl (Best regression model)")
    print("- final_combined_classification_model.pkl (Soil type classification model)")
    print("- feature_info.pkl (Feature column information for prediction)")
    
    print("\nYou can now run the Streamlit app with: streamlit run app.py")
