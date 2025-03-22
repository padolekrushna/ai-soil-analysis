import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from ai import engineer_features, prepare_input_for_prediction

# Set page config
st.set_page_config(page_title="Soil Analysis AI", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        regression_model = joblib.load("final_combined_regression_model.pkl")
        classification_model = joblib.load("final_combined_classification_model.pkl")
        feature_info = joblib.load("feature_info.pkl")
        return regression_model, classification_model, feature_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Function to make predictions
def predict_soil_properties(input_data):
    regression_model, classification_model, feature_info = load_models()
    
    if regression_model is None or classification_model is None or feature_info is None:
        return None
    
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

# App title
st.title("🌱 Soil Analysis AI")
st.write("Upload soil data or enter values manually to analyze soil properties")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Data", "Manual Entry", "Visualizations", "About"])

if page == "Home":
    st.write("""
    # Welcome to Soil Analysis AI
    
    This application uses machine learning to analyze soil properties and provide insights for agricultural decisions.
    
    ### Features:
    
    - **Soil Property Prediction**: Get accurate estimates of fertility, nutrients, and more
    - **Soil Type Classification**: Identify soil types based on measurements
    - **Visualized Results**: See your soil data through informative charts
    - **Recommendation Engine**: Receive tailored advice for soil improvement
    
    ### How to use:
    
    1. **Upload Data**: Use the Upload Data page to analyze multiple soil samples at once
    2. **Manual Entry**: Enter individual soil measurements for quick analysis
    3. **Visualizations**: Explore your soil data through charts and graphs
    """)
    
    st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2", caption="Soil Analysis", use_column_width=True)

elif page == "Upload Data":
    st.header("📊 Upload Soil Data")
    
    # Example CSV template download
    st.write("Download example template:")
    example_df = pd.DataFrame({
        "NIR_Spectroscopy_900nm": [1.5, 1.6],
        "NIR_Spectroscopy_2500nm": [2.5, 2.7],
        "Visible_Light_400nm": [3.0, 3.2],
        "Visible_Light_700nm": [4.0, 4.2],
        "Temperature_C": [25.0, 26.0],
        "Moisture_Content_%": [30.0, 32.0],
        "pH_Level": [6.5, 6.8],
        "Electrical_Conductivity_dS_m": [1.2, 1.3],
        "GPS_Latitude": [35.0, 35.1],
        "GPS_Longitude": [-90.0, -90.1],
        "Time_of_Measurement": [12, 14],
        "Nutrient_Nitrogen_mg_kg": [50.0, 52.0],
        "Nutrient_Phosphorus_mg_kg": [30.0, 32.0],
        "Nutrient_Potassium_mg_kg": [150.0, 155.0],
        "Organic_Matter_%": [3.0, 3.2]
    })
    csv = example_df.to_csv(index=False)
    st.download_button(
        label="Download Example CSV Template",
        data=csv,
        file_name="soil_data_template.csv",
        mime="text/csv",
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(input_data.head())
            
            if st.button("Analyze Soil"):
                with st.spinner("Analyzing soil data..."):
                    results_df = predict_soil_properties(input_data)
                    
                    if results_df is not None:
                        st.success("Analysis complete!")
                        st.write("Results:")
                        st.write(results_df)
                        
                        # Visualize key results
                        st.subheader("Fertility Score Distribution")
                        fig, ax = plt.subplots()
                        sns.histplot(results_df["Fertility_Score"], kde=True, ax=ax)
                        st.pyplot(fig)
                        
                        st.subheader("Nutrient Composition")
                        nutrient_cols = ["Nutrient_Nitrogen_mg_kg", "Nutrient_Phosphorus_mg_kg", "Nutrient_Potassium_mg_kg"]
                        avg_nutrients = results_df[nutrient_cols].mean()
                        fig, ax = plt.subplots()
                        sns.barplot(x=avg_nutrients.index, y=avg_nutrients.values, ax=ax)
                        ax.set_xticklabels(["Nitrogen", "Phosphorus", "Potassium"])
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="soil_analysis_results.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure your CSV file has all the required columns. Download the template for reference.")

elif page == "Manual Entry":
    st.header("🔍 Enter Soil Measurements")
    
    with st.form("soil_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nir_900 = st.number_input("NIR Spectroscopy 900nm", value=1.5)
            nir_2500 = st.number_input("NIR Spectroscopy 2500nm", value=2.5)
            vis_400 = st.number_input("Visible Light 400nm", value=3.0)
            vis_700 = st.number_input("Visible Light 700nm", value=4.0)
            temp = st.number_input("Temperature (°C)", value=25.0)
        
        with col2:
            moisture = st.number_input("Moisture Content (%)", value=30.0)
            ph = st.number_input("pH Level", value=6.5)
            ec = st.number_input("Electrical Conductivity (dS/m)", value=1.2)
            latitude = st.number_input("GPS Latitude", value=35.0)
            longitude = st.number_input("GPS Longitude", value=-90.0)
        
        with col3:
            time = st.number_input("Time of Measurement (hour)", value=12, min_value=0, max_value=23)
            nitrogen = st.number_input("Nitrogen (mg/kg)", value=50.0)
            phosphorus = st.number_input("Phosphorus (mg/kg)", value=30.0)
            potassium = st.number_input("Potassium (mg/kg)", value=150.0)
            organic = st.number_input("Organic Matter (%)", value=3.0)
        
        submit_button = st.form_submit_button("Analyze Soil")
    
    if submit_button:
        # Create a dataframe from the input
        input_data = pd.DataFrame({
            "NIR_Spectroscopy_900nm": [nir_900],
            "NIR_Spectroscopy_2500nm": [nir_2500],
            "Visible_Light_400nm": [vis_400],
            "Visible_Light_700nm": [vis_700],
            "Temperature_C": [temp],
            "Moisture_Content_%": [moisture],
            "pH_Level": [ph],
            "Electrical_Conductivity_dS_m": [ec],
            "GPS_Latitude": [latitude],
            "GPS_Longitude": [longitude],
            "Time_of_Measurement": [time],
            "Nutrient_Nitrogen_mg_kg": [nitrogen],
            "Nutrient_Phosphorus_mg_kg": [phosphorus],
            "Nutrient_Potassium_mg_kg": [potassium],
            "Organic_Matter_%": [organic]
        })
        
        with st.spinner("Analyzing soil data..."):
            results_df = predict_soil_properties(input_data)
            
            if results_df is not None:
                st.success("Analysis complete!")
                
                # Display results in a dashboard format
                st.subheader("Soil Analysis Results")
                
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fertility Score", f"{results_df['Fertility_Score'].iloc[0]:.2f}/10")
                    st.metric("Nitrogen (mg/kg)", f"{results_df['Nutrient_Nitrogen_mg_kg'].iloc[0]:.1f}")
                    st.metric("Phosphorus (mg/kg)", f"{results_df['Nutrient_Phosphorus_mg_kg'].iloc[0]:.1f}")
                
                with col2:
                    st.metric("Potassium (mg/kg)", f"{results_df['Nutrient_Potassium_mg_kg'].iloc[0]:.1f}")
                    st.metric("Organic Matter (%)", f"{results_df['Organic_Matter_%'].iloc[0]:.2f}")
                    st.metric("Water Retention", f"{results_df['Water_Retention_Capacity'].iloc[0]:.2f}")
                
                with col3:
                    st.metric("Lime Requirement", f"{results_df['Lime_Requirement'].iloc[0]:.2f}")
                    st.metric("Soil Erosion Risk", f"{results_df['Soil_Erosion_Risk'].iloc[0]:.2f}")
                    st.metric("Soil Type", f"{results_df['Soil_Type'].iloc[0]}")
                
                # NPK Bar Chart
                st.subheader("NPK Distribution")
                npk_data = {
                    'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
                    'Value': [
                        results_df['Nutrient_Nitrogen_mg_kg'].iloc[0],
                        results_df['Nutrient_Phosphorus_mg_kg'].iloc[0],
                        results_df['Nutrient_Potassium_mg_kg'].iloc[0]
                    ]
                }
                npk_df = pd.DataFrame(npk_data)
                fig, ax = plt.subplots()
                sns.barplot(x='Nutrient', y='Value', data=npk_df, ax=ax)
                st.pyplot(fig)
                
                # Recommendations based on analysis
                st.subheader("Recommendations")
                
                fertility = results_df['Fertility_Score'].iloc[0]
                if fertility < 5:
                    st.warning("Low fertility detected. Consider applying organic amendments and balanced fertilizers.")
                elif fertility < 7:
                    st.info("Moderate fertility. Regular fertilization and crop rotation recommended.")
                else:
                    st.success("Good fertility. Maintain current practices with routine soil testing.")
                
                # Nitrogen recommendations
                nitrogen = results_df['Nutrient_Nitrogen_mg_kg'].iloc[0]
                if nitrogen < 40:
                    st.warning("Low nitrogen. Consider legume cover crops and nitrogen fertilizers.")
                elif nitrogen > 80:
                    st.warning("High nitrogen. Reduce nitrogen applications to prevent leaching and runoff.")
                
                # pH recommendations
                if ph < 6:
                    st.warning(f"Acidic soil (pH {ph}). Consider adding lime to raise pH.")
                elif ph > 7.5:
                    st.warning(f"Alkaline soil (pH {ph}). Consider adding sulfur to lower pH.")

elif page == "Visualizations":
    st.header("📈 Soil Data Visualizations")
    
    st.write("""
    Upload a CSV file containing multiple soil samples to visualize trends and patterns.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file with multiple samples", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            
            if len(input_data) < 2:
                st.warning("Please upload a file with multiple soil samples for meaningful visualizations.")
            else:
                st.write(f"Loaded {len(input_data)} soil samples.")
                
                # Generate predictions
                with st.spinner("Analyzing soil data..."):
                    results_df = predict_soil_properties(input_data)
                    
                    if results_df is not None:
                        # Merge original data with predictions
                        if len(results_df) == len(input_data):
                            combined_df = pd.concat([input_data.reset_index(drop=True), 
                                                   results_df.reset_index(drop=True)], axis=1)
                        else:
                            combined_df = results_df
                        
                        # Visualization options
                        viz_type = st.selectbox(
                            "Select Visualization Type",
                            ["Fertility Distribution", "Nutrient Correlation", "Soil Type Distribution", 
                             "pH vs Fertility", "Custom Correlation"]
                        )
                        
                        if viz_type == "Fertility Distribution":
                            st.subheader("Fertility Score Distribution")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(results_df["Fertility_Score"], kde=True, ax=ax)
                            ax.set_xlabel("Fertility Score")
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                            
                            st.write(f"""
                            **Summary Statistics:**
                            - Mean: {results_df["Fertility_Score"].mean():.2f}
                            - Median: {results_df["Fertility_Score"].median():.2f}
                            - Min: {results_df["Fertility_Score"].min():.2f}
                            - Max: {results_df["Fertility_Score"].max():.2f}
                            """)
                        
                        elif viz_type == "Nutrient Correlation":
                            st.subheader("Nutrient Correlation Matrix")
                            nutrient_cols = ["Nutrient_Nitrogen_mg_kg", "Nutrient_Phosphorus_mg_kg", 
                                            "Nutrient_Potassium_mg_kg", "Organic_Matter_%"]
                            
                            corr = results_df[nutrient_cols].corr()
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                            st.pyplot(fig)
                        
                        elif viz_type == "Soil Type Distribution":
                            st.subheader("Soil Type Distribution")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            soil_counts = results_df["Soil_Type"].value_counts()
                            sns.barplot(x=soil_counts.index, y=soil_counts.values, ax=ax)
                            plt.xticks(rotation=45)
                            ax.set_xlabel("Soil Type")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
                            
                            # Show percentage distribution
                            st.write("Percentage Distribution:")
                            soil_percent = (soil_counts / soil_counts.sum() * 100).round(1)
                            for soil, pct in soil_percent.items():
                                st.write(f"- {soil}: {pct}%")
                        
                        elif viz_type == "pH vs Fertility":
                            st.subheader("pH Level vs Fertility Score")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            if "pH_Level" in combined_df.columns:
                                sns.scatterplot(x="pH_Level", y="Fertility_Score", data=combined_df, ax=ax)
                                ax.set_xlabel("pH Level")
                                ax.set_ylabel("Fertility Score")
                                
                                # Add regression line
                                sns.regplot(x="pH_Level", y="Fertility_Score", data=combined_df, 
                                           scatter=False, ax=ax, color='red')
                                
                                st.pyplot(fig)
                                
                                # Show optimal pH range
                                st.write("""
                                **Optimal pH Range:**
                                Most crops prefer a pH between 6.0 and 7.0. The chart shows how soil pH correlates with fertility.
                                """)
                            else:
                                st.error("pH_Level column not found in the dataset.")
                        
                        elif viz_type == "Custom Correlation":
                            st.subheader("Custom Correlation Analysis")
                            
                            # Get all numeric columns
                            numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                x_var = st.selectbox("Select X-axis variable", numeric_cols)
                            with col2:
                                y_var = st.selectbox("Select Y-axis variable", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                            
                            plot_type = st.radio("Select plot type", ["Scatter", "Scatter with trend", "Hexbin"])
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            if plot_type == "Scatter":
                                sns.scatterplot(x=x_var, y=y_var, data=combined_df, ax=ax)
                            elif plot_type == "Scatter with trend":
                                sns.regplot(x=x_var, y=y_var, data=combined_df, ax=ax)
                            elif plot_type == "Hexbin":
                                plt.hexbin(combined_df[x_var], combined_df[y_var], gridsize=20, cmap='Blues')
                                plt.colorbar(label='Count')
                            
                            ax.set_xlabel(x_var)
                            ax.set_ylabel(y_var)
                            st.pyplot(fig)
                            
                            # Calculate correlation
                            correlation = combined_df[[x_var, y_var]].corr().iloc[0, 1]
                            st.write(f"Correlation coefficient: {correlation:.3f}")
                            
                            if abs(correlation) > 0.7:
                                st.write("Strong correlation detected.")
                            elif abs(correlation) > 0.3:
                                st.write("Moderate correlation detected.")
                            else:
                                st.write("Weak correlation detected.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page == "About":
    st.header("ℹ️ About Soil Analysis AI")
    
    st.write("""
    ## How It Works
    
    This application uses advanced machine learning models to analyze soil properties and provide actionable insights for farmers and agricultural specialists.
    
    ### Machine Learning Models
    
    - **Random Forest**: Ensemble learning method for regression and classification
    - **XGBoost**: Gradient boosting framework known for performance and speed
    - **LightGBM**: Gradient boosting framework optimized for efficiency
    
    ### Features Used
    
    The models analyze various soil measurements including:
    - Near-infrared (NIR) spectroscopy readings at 900nm and 2500nm
    - Visible light readings at 400nm and 700nm
    - Temperature and moisture content
    - pH level and electrical conductivity
    - Geographic location and time of measurement
    - NPK nutrient levels and organic matter content
    
    ### Predictions
    
    The application provides predictions for:
    - Fertility Score
    - Nitrogen, Phosphorus, and Potassium levels
    - Organic Matter content
    - Water Retention Capacity
    - Lime Requirement
    - Soil Erosion Risk
    - Soil Type Classification
    """)
    
    st.write("""
    ## Data Privacy
    
    Your soil data is processed locally and is not stored on any server. All analyses are performed within your browser session.
    
    ## References
    
    This application is built on research in soil science, spectroscopy, and machine learning. For more information, please consult:
    
    - Soil Science Society of America Journal
    - Journal of Near Infrared Spectroscopy
    - Machine Learning applications in Agricultural Science
    """)

# Add footer
st.markdown("---")
st.markdown("© 2025 Soil Analysis AI | Developed for agricultural innovation")
