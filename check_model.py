import joblib

try:
    model = joblib.load('models/final_model.joblib')
    print("Model type:", type(model))
    if isinstance(model, dict):
        print("Model keys:", model.keys())
        for k in model.keys():
            print(f"Type of {k}:", type(model[k]))
            
except Exception as e:
    print(f"Error: {e}")
