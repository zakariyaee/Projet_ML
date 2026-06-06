import joblib
import pandas as pd
import json

try:
    model = joblib.load('models/final_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    print("Preprocessor type:", type(preprocessor))
    
    if hasattr(preprocessor, 'feature_names_in_'):
        print("Expected features (from preprocessor):")
        print(list(preprocessor.feature_names_in_))
    elif hasattr(model, 'feature_names_in_'):
        print("Expected features (from model):")
        print(list(model.feature_names_in_))
    else:
        print("Could not find feature_names_in_")
except Exception as e:
    print(f"Error: {e}")
