"""
Forest Fire Prediction Web Application
Streamlit-based interactive web app for forest fire predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import traceback

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import ml_pipeline with error handling
try:
    from ml_pipeline import ForestFirePredictor
except Exception as e:
    st.error(f"Error importing ml_pipeline: {str(e)}")
    st.stop()

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Get current theme
current_theme = st.session_state.get('theme', 'light')

# Custom CSS for better styling with theme support
# This CSS adapts to both light and dark themes
st.markdown(f"""
    <style>
    /* Main Header - works in both themes */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    /* Prediction Box - adapts to theme */
    .prediction-box {{
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }}
    
    /* Metric Card */
    .metric-card {{
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Light theme styles (default) */
    .stApp:not([data-theme="dark"]) .prediction-box,
    .prediction-box {{
        background-color: #f0f2f6;
    }}
    
    .stApp:not([data-theme="dark"]) .metric-card,
    .metric-card {{
        background-color: white;
    }}
    
    /* Dark theme styles */
    .stApp[data-theme="dark"] .prediction-box {{
        background-color: #1e1e1e;
        border-left-color: #FF6B35;
    }}
    
    .stApp[data-theme="dark"] .metric-card {{
        background-color: #262730;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    
    /* Force theme via class */
    .theme-light .prediction-box {{
        background-color: #f0f2f6 !important;
    }}
    
    .theme-light .metric-card {{
        background-color: white !important;
    }}
    
    .theme-dark .prediction-box {{
        background-color: #1e1e1e !important;
    }}
    
    .theme-dark .metric-card {{
        background-color: #262730 !important;
    }}
    
    /* Apply theme class to body */
    body.theme-{current_theme} {{
        /* Theme applied */
    }}
    
    /* Smooth transitions */
    .prediction-box, .metric-card {{
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state with error handling
try:
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ForestFirePredictor()
        st.session_state.models_loaded = False
        
        # Try to load models if they exist (check multiple locations)
        model_locations = ['models', '.']
        
        for location in model_locations:
            try:
                # Check if any model files exist
                reg_exists = (os.path.exists(f'{location}/regression_model.pkl') or 
                             os.path.exists(f'{location}/best_regression_model.pkl') or
                             os.path.exists('best_regression_model.pkl'))
                class_exists = (os.path.exists(f'{location}/classification_model.pkl') or 
                               os.path.exists(f'{location}/best_classification_model.pkl') or
                               os.path.exists('best_classification_model.pkl'))
                
                if reg_exists and class_exists:
                    try:
                        st.session_state.predictor.load_models(location if location != '.' else 'models')
                        st.session_state.models_loaded = True
                        break
                    except Exception as load_error:
                        # Silently continue if loading fails
                        continue
            except Exception as e:
                continue
except Exception as e:
    st.error(f"Error initializing predictor: {str(e)}")
    st.session_state.predictor = None
    st.session_state.models_loaded = False

# Apply theme to page via JavaScript
current_theme = st.session_state.get('theme', 'light')
st.markdown(f"""
<script>
(function() {{
    // Apply theme class to body and html
    document.documentElement.setAttribute('data-theme', '{current_theme}');
    document.documentElement.className = 'theme-{current_theme}';
    if (document.body) {{
        document.body.setAttribute('data-theme', '{current_theme}');
        document.body.className = 'theme-{current_theme}';
    }}
    
    // Also try to apply to Streamlit's main container
    const appContainer = document.querySelector('.stApp');
    if (appContainer) {{
        appContainer.setAttribute('data-theme', '{current_theme}');
        appContainer.className = 'stApp theme-{current_theme}';
    }}
}})();
</script>
""", unsafe_allow_html=True)

# Header - Always show this
try:
    st.markdown('<h1 class="main-header">🔥 Forest Fire Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
except Exception as e:
    st.title("🔥 Forest Fire Prediction System")
    st.markdown("---")

# Sidebar - Always show this
try:
    with st.sidebar:
        # Theme Toggle
        st.header("🎨 Theme")
        
        # Theme selector with visual feedback
        current_theme = st.session_state.get('theme', 'light')
        theme_options = ["🌞 Light Mode", "🌙 Dark Mode"]
        current_index = 0 if current_theme == 'light' else 1
        
        theme_choice = st.selectbox(
            "Select Theme:",
            theme_options,
            index=current_index,
            key="theme_selector"
        )
        
        # Update theme based on selection
        new_theme = 'light' if 'Light' in theme_choice else 'dark'
        if new_theme != current_theme:
            st.session_state.theme = new_theme
            st.rerun()  # Rerun to apply theme changes
        
        # Show current theme status
        if new_theme == 'light':
            st.success("🌞 Light mode active")
        else:
            st.info("🌙 Dark mode active")
        
        st.caption("💡 Theme changes apply immediately")
        
        st.markdown("---")
        
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Predict Fire", "📊 Model Info", "🔧 Train Models"],
            index=1  # Default to Model Info page
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This application predicts:
        - **Burned Area** (Regression)
        - **Fire Severity** (Classification)
        
        Using advanced ML models with feature engineering.
        """)
        
        # Show model status
        st.markdown("---")
        st.header("📊 Status")
        if st.session_state.get('models_loaded', False):
            st.success("✅ Models Loaded")
        else:
            st.warning("⚠️ Models Not Loaded")
            st.caption("Train models to make predictions")
except Exception as e:
    st.sidebar.error(f"Sidebar error: {str(e)}")
    page = "📊 Model Info"  # Default fallback

# Main content based on selected page
if page == "🏠 Predict Fire":
    st.header("Make a Prediction")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Models not found! Please train models first using the 'Train Models' page.")
        st.info("💡 Go to the '🔧 Train Models' page in the sidebar to train the models.")
        
        # Show input form even without models (for demonstration)
        st.markdown("### 📝 Input Form (Models need to be trained first)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Location & Time")
            X = st.number_input("X Coordinate", min_value=1, max_value=9, value=5, step=1)
            Y = st.number_input("Y Coordinate", min_value=1, max_value=9, value=5, step=1)
            
            month = st.selectbox(
                "Month",
                ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                index=7
            )
            
            day = st.selectbox(
                "Day of Week",
                ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],
                index=0
            )
        
        with col2:
            st.subheader("🌡️ Weather Conditions")
            temp = st.slider("Temperature (°C)", min_value=2.0, max_value=40.0, value=20.0, step=0.1)
            RH = st.slider("Relative Humidity (%)", min_value=15, max_value=100, value=50, step=1)
            wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
            rain = st.slider("Rainfall (mm)", min_value=0.0, max_value=7.0, value=0.0, step=0.1)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔥 Fire Weather Index")
            FFMC = st.slider("FFMC (Fine Fuel Moisture Code)", 
                            min_value=18.0, max_value=96.0, value=70.0, step=0.1)
            DMC = st.slider("DMC (Duff Moisture Code)", 
                           min_value=1.0, max_value=300.0, value=50.0, step=0.1)
        
        with col4:
            st.subheader("🌲 Fire Behavior Index")
            DC = st.slider("DC (Drought Code)", 
                          min_value=7.0, max_value=900.0, value=200.0, step=0.1)
            ISI = st.slider("ISI (Initial Spread Index)", 
                           min_value=0.0, max_value=60.0, value=10.0, step=0.1)
        
        st.stop()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location & Time")
        X = st.number_input(
            "Forest Grid - Horizontal Position", 
            min_value=1, 
            max_value=9, 
            value=5, 
            step=1,
            help="Horizontal position on the 9x9 forest grid map (1 = left, 9 = right)"
        )
        Y = st.number_input(
            "Forest Grid - Vertical Position", 
            min_value=1, 
            max_value=9, 
            value=5, 
            step=1,
            help="Vertical position on the 9x9 forest grid map (1 = bottom, 9 = top)"
        )
        
        month = st.selectbox(
            "Month of Year",
            ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
             'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            index=7,
            help="Select the month when the fire risk is being assessed"
        )
        
        day = st.selectbox(
            "Day of Week",
            ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],
            index=0,
            help="Select the day of the week"
        )
    
    with col2:
        st.subheader("🌡️ Weather Conditions")
        temp = st.slider(
            "Temperature", 
            min_value=2.0, 
            max_value=40.0, 
            value=20.0, 
            step=0.1,
            help="Air temperature in degrees Celsius (°C)"
        )
        RH = st.slider(
            "Relative Humidity", 
            min_value=15, 
            max_value=100, 
            value=50, 
            step=1,
            help="Relative humidity percentage (%) - how much moisture is in the air"
        )
        wind = st.slider(
            "Wind Speed", 
            min_value=0.0, 
            max_value=10.0, 
            value=4.0, 
            step=0.1,
            help="Wind speed in kilometers per hour (km/h)"
        )
        rain = st.slider(
            "Rainfall", 
            min_value=0.0, 
            max_value=7.0, 
            value=0.0, 
            step=0.1,
            help="Amount of rainfall in millimeters (mm) - 0 means no rain"
        )
    
    # Info about fire indices
    with st.expander("ℹ️ What are Fire Weather Indices? (Click to learn more)"):
        st.markdown("""
        **Fire Weather Indices** are scientific measurements used to assess fire risk:
        
        - **Fine Fuel Moisture (FFMC)**: How dry small materials like grass and leaves are
          - Low (18-50): Moist, less fire risk
          - Medium (50-75): Moderate dryness
          - High (75-96): Very dry, high fire risk
        
        - **Duff Moisture (DMC)**: Moisture in medium-sized fuels like twigs
          - Low (1-50): Moist conditions
          - Medium (50-100): Moderate dryness
          - High (100-300): Very dry, dangerous conditions
        
        - **Drought Code (DC)**: How dry deep organic matter is
          - Low (7-200): Normal conditions
          - Medium (200-400): Moderate drought
          - High (400-900): Severe drought, extreme fire risk
        
        - **Initial Spread Index (ISI)**: How fast a fire will spread
          - Low (0-5): Slow spread
          - Medium (5-15): Moderate spread
          - High (15-60): Very fast spread, dangerous
        """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🔥 Fire Weather Index")
        FFMC = st.slider(
            "Fine Fuel Moisture", 
            min_value=18.0, 
            max_value=96.0, 
            value=70.0, 
            step=0.1,
            help="How dry small fuels like grass and leaves are (18=damp, 96=very dry)"
        )
        DMC = st.slider(
            "Duff Moisture", 
            min_value=1.0, 
            max_value=300.0, 
            value=50.0, 
            step=0.1,
            help="Moisture in medium-sized fuels like twigs (1=wet, 300=very dry)"
        )
    
    with col4:
        st.subheader("🌲 Fire Behavior Index")
        DC = st.slider(
            "Drought Code", 
            min_value=7.0, 
            max_value=900.0, 
            value=200.0, 
            step=0.1,
            help="How dry deep organic matter is (7=normal, 900=severe drought)"
        )
        ISI = st.slider(
            "Initial Spread Index", 
            min_value=0.0, 
            max_value=60.0, 
            value=10.0, 
            step=0.1,
            help="How fast fire will spread initially (0=slow, 60=very fast)"
        )
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Fire Risk", type="primary", use_container_width=True):
        # Prepare input features
        features = {
            'X': X,
            'Y': Y,
            'month': month,
            'day': day,
            'FFMC': FFMC,
            'DMC': DMC,
            'DC': DC,
            'ISI': ISI,
            'temp': temp,
            'RH': RH,
            'wind': wind,
            'rain': rain
        }
        
        try:
            # Make predictions
            with st.spinner("Analyzing fire risk..."):
                if st.session_state.predictor is None:
                    st.error("Predictor not initialized. Please refresh the page.")
                    st.stop()
                
                # Regression prediction
                area_pred = st.session_state.predictor.predict_regression(features)
                
                # Classification prediction
                severity_class, severity_label = st.session_state.predictor.predict_classification(features)
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.header("📊 Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric(
                    label="🔥 Predicted Burned Area",
                    value=f"{area_pred:.2f} hectares",
                    delta=f"{'High' if area_pred > 25 else 'Medium' if area_pred > 1 else 'Low'} Risk"
                )
            
            with col_res2:
                # Color based on severity
                severity_colors = {
                    0: "🟢",
                    1: "🟠", 
                    2: "🔴"
                }
                st.metric(
                    label="⚠️ Fire Severity",
                    value=f"{severity_colors.get(severity_class, '⚪')} {severity_label}",
                    delta="Classification"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 💡 Insights")
            if area_pred < 1.0:
                st.success("✅ Low fire risk. Predicted area is minimal.")
            elif area_pred < 25.0:
                st.warning("⚠️ Moderate fire risk. Monitor conditions closely.")
            else:
                st.error("🚨 High fire risk! Predicted area is significant. Take immediate precautions.")
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                input_df = pd.DataFrame([features])
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please ensure all models are properly trained and loaded.")

elif page == "📊 Model Info":
    try:
        st.header("Model Information")
        
        st.markdown("""
        ### 🎯 Model Details
        
        This application uses two machine learning models:
        
        #### 1. **Regression Model** (Random Forest)
        - **Purpose**: Predicts the burned area in hectares
        - **Algorithm**: Random Forest Regressor with hyperparameter tuning
        - **Features**: 18 engineered features including cyclic encoding and interactions
        - **Output**: Continuous value (hectares)
        
        #### 2. **Classification Model** (Support Vector Machine)
        - **Purpose**: Classifies fire severity
        - **Algorithm**: SVM with RBF kernel
        - **Classes**: 
          - Small (< 1 ha)
          - Medium (1-25 ha)
          - Large (> 25 ha)
        - **Output**: Categorical classification
        
        ### 🔬 Feature Engineering
        
        The models use advanced feature engineering:
        - **Cyclic Encoding**: Month and day converted to sin/cos for circular patterns
        - **Interaction Features**: Temperature×Wind, Temperature×RH, FFMC×ISI, DMC×DC
        - **Log Transformation**: Applied to target variable for better distribution
        
        ### 📈 Model Performance
        
        - **Regression R²**: ~0.05-0.15 (expected for this dataset)
        - **Classification Accuracy**: ~50-51%
        - **Classification F1**: ~0.51 (weighted)
        
        *Note: Lower R² is expected due to dataset limitations (small size, weak correlations)*
        """)
        
        st.markdown("---")
        st.subheader("📦 Model Status")
        
        if st.session_state.get('models_loaded', False):
            st.success("✅ Models are loaded and ready for predictions!")
            st.info("💡 You can now make predictions on the '🏠 Predict Fire' page!")
        else:
            st.warning("⚠️ Models not loaded. Please train models first.")
            st.info("💡 Go to the '🔧 Train Models' page to train the models.")
            
            # Show instructions
            with st.expander("📖 How to Train Models"):
                st.markdown("""
                1. Go to the **🔧 Train Models** page in the sidebar
                2. Make sure `forestfires.csv` is in the project directory
                3. Click the **🚀 Train Models** button
                4. Wait for training to complete (may take a few minutes)
                5. Once trained, you can make predictions!
                """)
    except Exception as e:
        st.error(f"Error loading Model Info page: {str(e)}")
        st.code(traceback.format_exc())

elif page == "🔧 Train Models":
    st.header("Train Machine Learning Models")
    
    st.info("""
    This will train both regression and classification models using the forestfires.csv dataset.
    **Class Imbalance Handling**: Select a method to balance the dataset and improve predictions for minority classes.
    """)
    
    # Check if predictor is initialized
    if 'predictor' not in st.session_state or st.session_state.predictor is None:
        st.error("❌ Predictor not initialized. Please refresh the page.")
        st.stop()
    
    # Check if data file exists
    if not os.path.exists('forestfires.csv'):
        st.error("❌ forestfires.csv not found! Please ensure the data file is in the project directory.")
        st.info("💡 Make sure the forestfires.csv file is in the same directory as app.py")
        st.stop()
    
    # Class imbalance handling options
    st.markdown("### ⚖️ Class Imbalance Handling")
    
    # Check if imbalanced-learn is available
    try:
        import imblearn
        imblearn_available = True
    except ImportError:
        imblearn_available = False
        st.warning("⚠️ **imbalanced-learn not installed!** SMOTE and Undersampling require this package.")
        st.info("💡 Install with: `pip install imbalanced-learn`")
        st.info("💡 For now, only 'Class Weights' will be used.")
    
    col_balance1, col_balance2 = st.columns(2)
    
    with col_balance1:
        if imblearn_available:
            balance_method = st.selectbox(
                "Balancing Method:",
                ["SMOTE (Oversampling)", "Undersampling", "None (Original)"],
                index=0,
                help="SMOTE: Creates synthetic samples for minority classes\nUndersampling: Reduces majority class samples\nNone: Uses original imbalanced data"
            )
        else:
            balance_method = st.selectbox(
                "Balancing Method:",
                ["None (Original) - Install imbalanced-learn for SMOTE/Undersampling"],
                index=0,
                disabled=True,
                help="Install imbalanced-learn to use SMOTE or Undersampling"
            )
        
        # Convert to method name
        if "SMOTE" in balance_method:
            method = "smote"
        elif "Undersampling" in balance_method:
            method = "undersample"
        else:
            method = "none"
    
    with col_balance2:
        use_class_weight = st.checkbox(
            "Use Class Weights",
            value=True,
            help="Automatically adjust class weights to give more importance to minority classes"
        )
    
    st.markdown("---")
    
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Capture output for display
                import sys
                from io import StringIO
                
                # Redirect stdout to capture print statements
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                # Train models
                results = st.session_state.predictor.train_models(
                    'forestfires.csv',
                    balance_method=method,
                    use_class_weight=use_class_weight
                )
                
                # Restore stdout
                sys.stdout = old_stdout
                training_output = captured_output.getvalue()
                
                # Save models
                st.session_state.predictor.save_models('models')
                
                # Reload models
                try:
                    st.session_state.predictor.load_models('models')
                    st.session_state.models_loaded = True
                except Exception as load_err:
                    st.warning(f"Models trained but loading failed: {str(load_err)}")
                    st.session_state.models_loaded = False
                
                st.success("✅ Models trained and saved successfully!")
                
                # Display training output
                with st.expander("📋 Detailed Training Output", expanded=True):
                    st.text(training_output)
                
                # Display results
                st.markdown("### 📊 Training Results Summary")
                
                col_train1, col_train2 = st.columns(2)
                
                with col_train1:
                    st.markdown("#### Regression Model")
                    st.metric(
                        label="R² Score",
                        value=f"{results['regression']['r2']:.4f}"
                    )
                    st.metric(
                        label="RMSE",
                        value=f"{results['regression']['rmse']:.4f}"
                    )
                
                with col_train2:
                    st.markdown("#### Classification Model")
                    st.metric(
                        label="Accuracy",
                        value=f"{results['classification']['accuracy']:.4f}"
                    )
                    st.metric(
                        label="F1 Score (Weighted)",
                        value=f"{results['classification']['f1_weighted']:.4f}"
                    )
                
                # Before/After Comparison
                if 'baseline' in results:
                    st.markdown("### 📈 Before/After Comparison")
                    
                    comparison_data = {
                        'Metric': ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'Precision', 'Recall'],
                        'Before (Baseline)': [
                            results['baseline']['accuracy'],
                            results['baseline']['f1_weighted'],
                            results['baseline']['f1_macro'],
                            results['baseline']['precision'],
                            results['baseline']['recall']
                        ],
                        'After (Balanced)': [
                            results['classification']['accuracy'],
                            results['classification']['f1_weighted'],
                            results['classification']['f1_macro'],
                            results['classification']['precision'],
                            results['classification']['recall']
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Improvement'] = comparison_df['After (Balanced)'] - comparison_df['Before (Baseline)']
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Visual comparison
                    col_comp1, col_comp2 = st.columns(2)
                    
                    with col_comp1:
                        st.metric(
                            "Accuracy Improvement",
                            f"{comparison_df.loc[0, 'Improvement']:+.4f}",
                            delta=f"{comparison_df.loc[0, 'Improvement']/results['baseline']['accuracy']*100:+.1f}%"
                        )
                    
                    with col_comp2:
                        st.metric(
                            "F1 Score Improvement",
                            f"{comparison_df.loc[1, 'Improvement']:+.4f}",
                            delta=f"{comparison_df.loc[1, 'Improvement']/results['baseline']['f1_weighted']*100:+.1f}%"
                        )
                
                # Per-class metrics
                if 'f1_per_class' in results['classification']:
                    st.markdown("### 🎯 Per-Class Performance")
                    
                    per_class_data = {
                        'Class': ['Small', 'Medium', 'Large'],
                        'Precision': results['classification']['precision_per_class'],
                        'Recall': results['classification']['recall_per_class'],
                        'F1-Score': results['classification']['f1_per_class']
                    }
                    
                    per_class_df = pd.DataFrame(per_class_data)
                    st.dataframe(per_class_df, use_container_width=True, hide_index=True)
                
                # Confusion Matrix
                if 'confusion_matrix' in results['classification']:
                    st.markdown("### 🔢 Confusion Matrix")
                    cm = np.array(results['classification']['confusion_matrix'])
                    cm_df = pd.DataFrame(
                        cm,
                        index=['Actual Small', 'Actual Medium', 'Actual Large'],
                        columns=['Predicted Small', 'Predicted Medium', 'Predicted Large']
                    )
                    st.dataframe(cm_df, use_container_width=True)
                
                st.balloons()
                
                # Show next steps
                st.info("💡 You can now go to the '🏠 Predict Fire' page to make predictions!")
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                st.code(traceback.format_exc())
                st.info("Please check the error message above and ensure all dependencies are installed.")
                st.info("💡 Make sure you have installed: `pip install imbalanced-learn`")

# Footer - Always show
try:
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Forest Fire Prediction System | Built with Streamlit & Scikit-learn"
        "</div>",
        unsafe_allow_html=True
    )
except:
    st.markdown("---")
    st.caption("Forest Fire Prediction System | Built with Streamlit & Scikit-learn")

