# 🔥 Forest Fire Prediction System

A comprehensive machine learning application for predicting forest fire burned area and severity classification using advanced feature engineering and multiple ML models.

## 📋 Features

- **Regression Model**: Predicts burned area in hectares using Random Forest
- **Classification Model**: Classifies fire severity (Small/Medium/Large) using SVM
- **Advanced Feature Engineering**: Cyclic encoding, interaction features, log transformations
- **Interactive Web Interface**: Streamlit-based user-friendly UI
- **Model Training**: Built-in training pipeline with hyperparameter tuning

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset**:
   - Make sure `forestfires.csv` is in the project root directory

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**:
   - The app will automatically open at `http://localhost:8501`

## 📦 Project Structure

```
.
├── app.py                      # Streamlit web application
├── ml_pipeline.py              # Core ML pipeline and prediction logic
├── forest_fire_FINAL.ipynb     # Jupyter notebook with full analysis
├── forestfires.csv             # Dataset (required)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Dockerfile                  # Docker configuration
├── .gitignore                  # Git ignore rules
└── models/                     # Saved models directory (created after training)
    ├── regression_model.pkl
    ├── classification_model.pkl
    ├── scaler_regression.pkl
    └── scaler_classification.pkl
```

## 🎯 Usage

### Option 1: Use Pre-trained Models

If you have models saved from the notebook:
1. Run the notebook to generate models (saved as `best_regression_model.pkl`, etc.)
2. Use the conversion script to rename them:
   ```bash
   python convert_models.py
   ```
3. Start the app and use the "Predict Fire" page

### Option 2: Train Models via Web App

1. Start the Streamlit app
2. Navigate to "🔧 Train Models" page
3. Click "🚀 Train Models" button
4. Wait for training to complete (may take a few minutes)
5. Models will be automatically saved and loaded

### Making Predictions

1. Go to "🏠 Predict Fire" page
2. Enter the required parameters:
   - Location (X, Y coordinates)
   - Date (Month, Day)
   - Weather conditions (Temperature, Humidity, Wind, Rain)
   - Fire indices (FFMC, DMC, DC, ISI)
3. Click "🔮 Predict Fire Risk"
4. View predictions for:
   - **Burned Area** (in hectares)
   - **Fire Severity** (Small/Medium/Large)

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t forest-fire-prediction .
```

### Run Docker Container

```bash
docker run -p 8501:8501 forest-fire-prediction
```

The app will be available at `http://localhost:8501`

## ☁️ Cloud Deployment

### Heroku

1. **Install Heroku CLI** (if not already installed)

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create a new app**:
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

5. **Open the app**:
   ```bash
   heroku open
   ```

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Other Platforms

- **AWS EC2**: Use the Dockerfile or deploy directly
- **Google Cloud Run**: Use the Dockerfile
- **Azure App Service**: Use the Dockerfile
- **Railway**: Connect GitHub repo and deploy

## 🔧 Configuration

### Model Parameters

You can modify model parameters in `ml_pipeline.py`:

- **Regression Model**: Random Forest hyperparameters in `train_models()` method
- **Classification Model**: SVM parameters (C, gamma, kernel)

### Feature Engineering

Feature engineering logic is in `ml_pipeline.py` in the `engineer_features()` method. You can add or modify features there.

## 📊 Model Performance

- **Regression R²**: ~0.05-0.15 (expected for this dataset)
- **Classification Accuracy**: ~50-51%
- **Classification F1**: ~0.51 (weighted)

*Note: Lower R² is expected due to dataset limitations (small size, weak correlations)*

## 🛠️ Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Structure

- `app.py`: Streamlit UI and user interface
- `ml_pipeline.py`: Core ML logic, training, and prediction
- `forest_fire_FINAL.ipynb`: Complete analysis and model development

## 📝 Requirements

See `requirements.txt` for full list. Main dependencies:

- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available for educational purposes.

## ⚠️ Important Notes

- The dataset is small (517 samples), which limits model performance
- Models are trained for demonstration purposes
- For production use, consider:
  - Larger, more comprehensive datasets
  - Additional feature engineering
  - Model ensemble techniques
  - Regular model retraining

## 🆘 Troubleshooting

### Models Not Found Error

- Ensure models are trained first (use "Train Models" page)
- Or run the notebook and convert models using `convert_models.py`

### Import Errors

- Make sure all dependencies are installed: `pip install -r requirements.txt`

### Port Already in Use

- Change the port: `streamlit run app.py --server.port 8502`

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ using Streamlit and Scikit-learn**

