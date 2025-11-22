"""
Script to convert models from notebook naming convention to app naming convention
Run this after training models in the notebook
"""

import os
import shutil
import joblib

def convert_models():
    """Convert notebook models to app format"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Mapping from notebook names to app names
    model_mapping = {
        'best_regression_model.pkl': 'regression_model.pkl',
        'best_classification_model.pkl': 'classification_model.pkl',
        'scaler_regression.pkl': 'scaler_regression.pkl',
        'scaler_classification.pkl': 'scaler_classification.pkl'
    }
    
    converted = []
    missing = []
    
    print("Converting models from notebook format to app format...")
    print("=" * 60)
    
    for old_name, new_name in model_mapping.items():
        if os.path.exists(old_name):
            # Copy to models directory with new name
            shutil.copy(old_name, f'models/{new_name}')
            converted.append(old_name)
            print(f"✓ Converted: {old_name} -> models/{new_name}")
        else:
            missing.append(old_name)
            print(f"✗ Not found: {old_name}")
    
    print("=" * 60)
    
    if converted:
        print(f"\n✅ Successfully converted {len(converted)} model(s)")
        print("Models are now ready for the Streamlit app!")
    else:
        print("\n⚠️  No models found to convert.")
        print("Please train models first using the notebook or the web app.")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} model file(s).")
        print("Make sure you've run the notebook completely to generate all models.")

if __name__ == '__main__':
    convert_models()

