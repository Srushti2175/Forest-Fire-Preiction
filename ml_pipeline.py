"""
Forest Fire Prediction ML Pipeline
Core machine learning functions for training and prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix, 
                             classification_report, precision_score, recall_score)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE and undersampling will not be available.")
    print("Install with: pip install imbalanced-learn")

from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')


class ForestFirePredictor:
    """Main class for forest fire prediction"""
    
    def __init__(self):
        self.scaler_reg = None
        self.scaler_class = None
        self.regression_model = None
        self.classification_model = None
        self.feature_columns = None
        
    def load_data(self, csv_path='forestfires.csv'):
        """Load and return the dataset"""
        return pd.read_csv(csv_path)
    
    def engineer_features(self, df):
        """Apply feature engineering transformations"""
        processed_data = df.copy()
        
        # Month/day mappings
        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
        
        processed_data['month_num'] = processed_data['month'].map(month_mapping)
        processed_data['day_num'] = processed_data['day'].map(day_mapping)
        
        # Cyclic encoding
        processed_data['month_sin'] = np.sin(2 * np.pi * processed_data['month_num'] / 12)
        processed_data['month_cos'] = np.cos(2 * np.pi * processed_data['month_num'] / 12)
        processed_data['day_sin'] = np.sin(2 * np.pi * processed_data['day_num'] / 7)
        processed_data['day_cos'] = np.cos(2 * np.pi * processed_data['day_num'] / 7)
        
        # Interaction features
        processed_data['temp_wind'] = processed_data['temp'] * processed_data['wind']
        processed_data['temp_RH'] = processed_data['temp'] * processed_data['RH']
        processed_data['FFMC_ISI'] = processed_data['FFMC'] * processed_data['ISI']
        processed_data['DMC_DC'] = processed_data['DMC'] * processed_data['DC']
        
        # Log transform
        processed_data['logArea'] = np.log1p(processed_data['area'])
        
        # Fire severity classification
        def classify_fire_severity(area):
            if area < 1.0:
                return 0  # Small
            elif area < 25.0:
                return 1  # Medium
            else:
                return 2  # Large
        
        processed_data['fireSeverity'] = processed_data['area'].apply(classify_fire_severity)
        
        return processed_data
    
    def prepare_features(self, processed_data):
        """Define and extract feature columns"""
        self.feature_columns = [
            'X', 'Y', 
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'FFMC', 'DMC', 'DC', 'ISI', 
            'temp', 'RH', 'wind', 'rain',
            'temp_wind', 'temp_RH', 'FFMC_ISI', 'DMC_DC'
        ]
        
        # Regression: Remove zero-area cases
        regression_data = processed_data[processed_data['area'] > 0].copy()
        x_regression = regression_data[self.feature_columns]
        y_regression = regression_data['logArea']
        
        # Classification: Use all data
        x_classification = processed_data[self.feature_columns]
        y_classification = processed_data['fireSeverity']
        
        return (x_regression, y_regression, x_classification, y_classification)
    
    def train_models(self, csv_path='forestfires.csv', random_seed=42, 
                     balance_method='smote', use_class_weight=True):
        """Train both regression and classification models with class imbalance handling
        
        Args:
            csv_path: Path to CSV file
            random_seed: Random seed for reproducibility
            balance_method: Method to balance classes - 'smote', 'undersample', 'none'
            use_class_weight: Whether to use class weights in the model
        
        Returns:
            Dictionary with training results including before/after metrics
        """
        # Load and process data
        df = self.load_data(csv_path)
        processed_data = self.engineer_features(df)
        x_reg, y_reg, x_class, y_class = self.prepare_features(processed_data)
        
        # Train-test split
        x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
            x_reg, y_reg, test_size=0.2, random_state=random_seed
        )
        
        x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
            x_class, y_class, test_size=0.2, random_state=random_seed, stratify=y_class
        )
        
        # Scale features
        self.scaler_reg = StandardScaler()
        self.scaler_class = StandardScaler()
        
        x_train_reg_scaled = self.scaler_reg.fit_transform(x_train_reg)
        x_test_reg_scaled = self.scaler_reg.transform(x_test_reg)
        
        x_train_class_scaled = self.scaler_class.fit_transform(x_train_class)
        x_test_class_scaled = self.scaler_class.transform(x_test_class)
        
        # Train regression model with hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_model = RandomForestRegressor(random_state=random_seed)
        grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(x_train_reg_scaled, y_train_reg)
        
        self.regression_model = grid_search.best_estimator_
        
        # ===== CLASS IMBALANCE HANDLING =====
        # Show class distribution BEFORE balancing
        class_dist_before = Counter(y_train_class)
        print("\n" + "="*70)
        print("CLASS DISTRIBUTION - BEFORE BALANCING")
        print("="*70)
        for class_label, count in sorted(class_dist_before.items()):
            class_name = {0: 'Small', 1: 'Medium', 2: 'Large'}[class_label]
            print(f"  {class_name}: {count} samples ({count/len(y_train_class)*100:.1f}%)")
        
        # Train baseline model (no balancing) for comparison
        baseline_model = SVC(kernel='rbf', C=10, gamma='scale', 
                         random_state=random_seed, class_weight=None)
        baseline_model.fit(x_train_class_scaled, y_train_class)
        baseline_pred = baseline_model.predict(x_test_class_scaled)
        baseline_metrics = {
            'accuracy': accuracy_score(y_test_class, baseline_pred),
            'f1_weighted': f1_score(y_test_class, baseline_pred, average='weighted'),
            'f1_macro': f1_score(y_test_class, baseline_pred, average='macro'),
            'precision': precision_score(y_test_class, baseline_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_class, baseline_pred, average='weighted', zero_division=0)
        }
        
        # Apply balancing method
        x_train_balanced = x_train_class_scaled.copy()
        y_train_balanced = y_train_class.copy()
        balance_info = ""
        
        if balance_method == 'smote':
            if not IMBALANCED_LEARN_AVAILABLE:
                print("\n" + "="*70)
                print("ERROR: SMOTE requires imbalanced-learn package")
                print("="*70)
                print("Please install with: pip install imbalanced-learn")
                print("Falling back to class weights only...")
                balance_method = 'none'
                x_train_balanced = x_train_class_scaled.copy()
                y_train_balanced = y_train_class.copy()
            else:
                print("\n" + "="*70)
                print("APPLYING SMOTE OVERSAMPLING...")
                print("="*70)
                try:
                    # Calculate safe k_neighbors value
                    min_samples = min(class_dist_before.values())
                    k_neighbors = min(5, max(1, min_samples - 1))
                    
                    smote = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
                    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_class_scaled, y_train_class)
                    class_dist_after = Counter(y_train_balanced)
                    balance_info = "SMOTE Oversampling"
                    print("Class distribution AFTER SMOTE:")
                    for class_label, count in sorted(class_dist_after.items()):
                        class_name = {0: 'Small', 1: 'Medium', 2: 'Large'}[class_label]
                        print(f"  {class_name}: {count} samples ({count/len(y_train_balanced)*100:.1f}%)")
                except Exception as e:
                    print(f"SMOTE failed: {e}. Using original data.")
                    balance_method = 'none'
                    x_train_balanced = x_train_class_scaled.copy()
                    y_train_balanced = y_train_class.copy()
        
        elif balance_method == 'undersample':
            if not IMBALANCED_LEARN_AVAILABLE:
                print("\n" + "="*70)
                print("ERROR: Undersampling requires imbalanced-learn package")
                print("="*70)
                print("Please install with: pip install imbalanced-learn")
                print("Falling back to class weights only...")
                balance_method = 'none'
                x_train_balanced = x_train_class_scaled.copy()
                y_train_balanced = y_train_class.copy()
            else:
                print("\n" + "="*70)
                print("APPLYING RANDOM UNDERSAMPLING...")
                print("="*70)
                try:
                    rus = RandomUnderSampler(random_state=random_seed)
                    x_train_balanced, y_train_balanced = rus.fit_resample(x_train_class_scaled, y_train_class)
                    class_dist_after = Counter(y_train_balanced)
                    balance_info = "Random Undersampling"
                    print("Class distribution AFTER Undersampling:")
                    for class_label, count in sorted(class_dist_after.items()):
                        class_name = {0: 'Small', 1: 'Medium', 2: 'Large'}[class_label]
                        print(f"  {class_name}: {count} samples ({count/len(y_train_balanced)*100:.1f}%)")
                except Exception as e:
                    print(f"Undersampling failed: {e}. Using original data.")
                    balance_method = 'none'
                    x_train_balanced = x_train_class_scaled.copy()
                    y_train_balanced = y_train_class.copy()
        
        else:
            balance_info = "No Balancing"
            print("\n" + "="*70)
            print("NO BALANCING APPLIED (Using original distribution)")
            print("="*70)
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train_class)
            weights = compute_class_weight('balanced', classes=classes, y=y_train_class)
            class_weights = dict(zip(classes, weights))
            print(f"\nClass weights: {class_weights}")
            balance_info += " + Class Weights"
        
        # Train classification model with balancing
        if use_class_weight and class_weights:
            self.classification_model = SVC(kernel='rbf', C=10, gamma='scale', 
                                           random_state=random_seed, class_weight=class_weights)
        else:
            self.classification_model = SVC(kernel='rbf', C=10, gamma='scale', 
                                           random_state=random_seed, class_weight='balanced')
        
        self.classification_model.fit(x_train_balanced, y_train_balanced)
        
        # Evaluate models
        reg_pred = self.regression_model.predict(x_test_reg_scaled)
        reg_r2 = r2_score(y_test_reg, reg_pred)
        reg_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))
        
        class_pred = self.classification_model.predict(x_test_class_scaled)
        class_acc = accuracy_score(y_test_class, class_pred)
        class_f1_weighted = f1_score(y_test_class, class_pred, average='weighted')
        class_f1_macro = f1_score(y_test_class, class_pred, average='macro')
        class_precision = precision_score(y_test_class, class_pred, average='weighted', zero_division=0)
        class_recall = recall_score(y_test_class, class_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_f1_per_class = f1_score(y_test_class, class_pred, average=None, zero_division=0)
        class_precision_per_class = precision_score(y_test_class, class_pred, average=None, zero_division=0)
        class_recall_per_class = recall_score(y_test_class, class_pred, average=None, zero_division=0)
        
        # Print comparison
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        print(f"\n{'Metric':<20} {'Before (Baseline)':<20} {'After ({balance_info})':<20} {'Improvement':<15}")
        print("-"*70)
        print(f"{'Accuracy':<20} {baseline_metrics['accuracy']:<20.4f} {class_acc:<20.4f} {class_acc - baseline_metrics['accuracy']:+.4f}")
        print(f"{'F1 (Weighted)':<20} {baseline_metrics['f1_weighted']:<20.4f} {class_f1_weighted:<20.4f} {class_f1_weighted - baseline_metrics['f1_weighted']:+.4f}")
        print(f"{'F1 (Macro)':<20} {baseline_metrics['f1_macro']:<20.4f} {class_f1_macro:<20.4f} {class_f1_macro - baseline_metrics['f1_macro']:+.4f}")
        print(f"{'Precision':<20} {baseline_metrics['precision']:<20.4f} {class_precision:<20.4f} {class_precision - baseline_metrics['precision']:+.4f}")
        print(f"{'Recall':<20} {baseline_metrics['recall']:<20.4f} {class_recall:<20.4f} {class_recall - baseline_metrics['recall']:+.4f}")
        
        print("\n" + "="*70)
        print("PER-CLASS METRICS (After Balancing)")
        print("="*70)
        print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
        print("-"*70)
        for i, class_name in enumerate(['Small', 'Medium', 'Large']):
            print(f"{class_name:<15} {class_precision_per_class[i]:<15.4f} {class_recall_per_class[i]:<15.4f} {class_f1_per_class[i]:<15.4f}")
        
        print("\n" + "="*70)
        print("CONFUSION MATRIX (After Balancing)")
        print("="*70)
        cm = confusion_matrix(y_test_class, class_pred)
        print(f"\n{'':<10} {'Predicted Small':<15} {'Predicted Medium':<15} {'Predicted Large':<15}")
        for i, class_name in enumerate(['Actual Small', 'Actual Medium', 'Actual Large']):
            print(f"{class_name:<10} {cm[i][0]:<15} {cm[i][1]:<15} {cm[i][2]:<15}")
        
        return {
            'regression': {'r2': reg_r2, 'rmse': reg_rmse},
            'classification': {
                'accuracy': class_acc, 
                'f1_weighted': class_f1_weighted,
                'f1_macro': class_f1_macro,
                'precision': class_precision,
                'recall': class_recall,
                'f1_per_class': class_f1_per_class.tolist(),
                'precision_per_class': class_precision_per_class.tolist(),
                'recall_per_class': class_recall_per_class.tolist(),
                'confusion_matrix': cm.tolist()
            },
            'baseline': baseline_metrics,
            'balance_method': balance_info,
            'class_distribution_before': dict(class_dist_before),
            'class_distribution_after': dict(Counter(y_train_balanced)) if balance_method != 'none' else dict(class_dist_before)
        }
    
    def predict_regression(self, features_dict):
        """Predict burned area (in log scale)"""
        if self.regression_model is None or self.scaler_reg is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Engineer features for single prediction
        processed = self._engineer_single_features(input_df)
        
        # Extract features
        features = processed[self.feature_columns].values
        
        # Scale and predict
        features_scaled = self.scaler_reg.transform(features)
        log_area_pred = self.regression_model.predict(features_scaled)[0]
        
        # Convert back to original scale
        area_pred = np.expm1(log_area_pred)
        
        return max(0, area_pred)  # Ensure non-negative
    
    def predict_classification(self, features_dict):
        """Predict fire severity class"""
        if self.classification_model is None or self.scaler_class is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Engineer features for single prediction
        processed = self._engineer_single_features(input_df)
        
        # Extract features
        features = processed[self.feature_columns].values
        
        # Scale and predict
        features_scaled = self.scaler_class.transform(features)
        severity = self.classification_model.predict(features_scaled)[0]
        
        severity_map = {0: 'Small (< 1 ha)', 1: 'Medium (1-25 ha)', 2: 'Large (> 25 ha)'}
        
        return severity, severity_map[severity]
    
    def _engineer_single_features(self, df):
        """Engineer features for a single prediction"""
        processed = df.copy()
        
        # Month/day mappings
        month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
        
        if 'month' in processed.columns:
            processed['month_num'] = processed['month'].map(month_mapping)
        if 'day' in processed.columns:
            processed['day_num'] = processed['day'].map(day_mapping)
        
        # Cyclic encoding
        if 'month_num' in processed.columns:
            processed['month_sin'] = np.sin(2 * np.pi * processed['month_num'] / 12)
            processed['month_cos'] = np.cos(2 * np.pi * processed['month_num'] / 12)
        if 'day_num' in processed.columns:
            processed['day_sin'] = np.sin(2 * np.pi * processed['day_num'] / 7)
            processed['day_cos'] = np.cos(2 * np.pi * processed['day_num'] / 7)
        
        # Interaction features
        if 'temp' in processed.columns and 'wind' in processed.columns:
            processed['temp_wind'] = processed['temp'] * processed['wind']
        if 'temp' in processed.columns and 'RH' in processed.columns:
            processed['temp_RH'] = processed['temp'] * processed['RH']
        if 'FFMC' in processed.columns and 'ISI' in processed.columns:
            processed['FFMC_ISI'] = processed['FFMC'] * processed['ISI']
        if 'DMC' in processed.columns and 'DC' in processed.columns:
            processed['DMC_DC'] = processed['DMC'] * processed['DC']
        
        return processed
    
    def save_models(self, model_dir='models'):
        """Save trained models and scalers"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.regression_model:
            joblib.dump(self.regression_model, f'{model_dir}/regression_model.pkl')
        if self.classification_model:
            joblib.dump(self.classification_model, f'{model_dir}/classification_model.pkl')
        if self.scaler_reg:
            joblib.dump(self.scaler_reg, f'{model_dir}/scaler_regression.pkl')
        if self.scaler_class:
            joblib.dump(self.scaler_class, f'{model_dir}/scaler_classification.pkl')
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir='models'):
        """Load pre-trained models and scalers
        
        Supports both naming conventions:
        - App format: regression_model.pkl, classification_model.pkl
        - Notebook format: best_regression_model.pkl, best_classification_model.pkl
        """
        import os
        
        # Try app naming convention first, then notebook convention
        reg_model_path = None
        class_model_path = None
        
        # Regression model
        if os.path.exists(f'{model_dir}/regression_model.pkl'):
            reg_model_path = f'{model_dir}/regression_model.pkl'
        elif os.path.exists(f'{model_dir}/best_regression_model.pkl'):
            reg_model_path = f'{model_dir}/best_regression_model.pkl'
        elif os.path.exists('best_regression_model.pkl'):
            reg_model_path = 'best_regression_model.pkl'
        else:
            raise FileNotFoundError("Regression model not found. Please train models first.")
        
        # Classification model
        if os.path.exists(f'{model_dir}/classification_model.pkl'):
            class_model_path = f'{model_dir}/classification_model.pkl'
        elif os.path.exists(f'{model_dir}/best_classification_model.pkl'):
            class_model_path = f'{model_dir}/best_classification_model.pkl'
        elif os.path.exists('best_classification_model.pkl'):
            class_model_path = 'best_classification_model.pkl'
        else:
            raise FileNotFoundError("Classification model not found. Please train models first.")
        
        # Scalers
        scaler_reg_path = None
        scaler_class_path = None
        
        if os.path.exists(f'{model_dir}/scaler_regression.pkl'):
            scaler_reg_path = f'{model_dir}/scaler_regression.pkl'
        elif os.path.exists('scaler_regression.pkl'):
            scaler_reg_path = 'scaler_regression.pkl'
        else:
            raise FileNotFoundError("Regression scaler not found.")
        
        if os.path.exists(f'{model_dir}/scaler_classification.pkl'):
            scaler_class_path = f'{model_dir}/scaler_classification.pkl'
        elif os.path.exists('scaler_classification.pkl'):
            scaler_class_path = 'scaler_classification.pkl'
        else:
            raise FileNotFoundError("Classification scaler not found.")
        
        # Load models and scalers
        self.regression_model = joblib.load(reg_model_path)
        self.classification_model = joblib.load(class_model_path)
        self.scaler_reg = joblib.load(scaler_reg_path)
        self.scaler_class = joblib.load(scaler_class_path)
        
        # Set feature columns (needed for prediction)
        self.feature_columns = [
            'X', 'Y', 
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'FFMC', 'DMC', 'DC', 'ISI', 
            'temp', 'RH', 'wind', 'rain',
            'temp_wind', 'temp_RH', 'FFMC_ISI', 'DMC_DC'
        ]
        
        print(f"Models loaded successfully!")
        print(f"  Regression: {reg_model_path}")
        print(f"  Classification: {class_model_path}")

