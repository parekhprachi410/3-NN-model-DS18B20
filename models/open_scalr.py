import joblib

# Load the scaler
scaler = joblib.load("scaler.save")

# Get the min and scale values
print("Scaler type:", type(scaler).__name__)
print("Data min:", scaler.data_min_)
print("Data max:", scaler.data_max_)
print("Data range:", scaler.data_range_)
print("Min parameter:", scaler.min_)
print("Scale parameter:", scaler.scale_)
