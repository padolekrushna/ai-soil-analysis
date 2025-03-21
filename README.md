# Soil Analysis Streamlit App Deployment

## Local Development Setup

### 1. Clone the Repository
```bash
git clone your-repo-url
cd soil-analysis-app
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place Model Files
Ensure these files are in the root directory:
- `final_combined_regression_model.pkl`
- `final_combined_classification_model.pkl`

### 5. Run Streamlit App
```bash
streamlit run app.py
```

## Cloud Deployment Options

### Streamlit Cloud
1. Push code to GitHub
2. Connect Streamlit Cloud to your repository
3. Select `app.py` as the main file

### Heroku Deployment
```bash
heroku create soil-analysis-streamlit
heroku buildpacks:set heroku/python
git push heroku main
```

### Google Cloud Run
1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD streamlit run --server.port 8501 --server.address 0.0.0.0 app.py
```

## Troubleshooting
- Ensure all dependencies are installed
- Check model file paths
- Verify input feature names match model expectations

## Model Maintenance
- Retrain models periodically
- Monitor prediction performance
- Update requirements as needed
