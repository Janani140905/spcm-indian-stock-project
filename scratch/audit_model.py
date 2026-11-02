import sys
import joblib

sys.path.insert(0, ".")
from src.features.price_features_inference import ModelArtifactBundle

bundle = joblib.load("models/spcm_model_bundle.joblib")
print("Model Name:", bundle.model_name)
print("Model Type:", type(bundle.model).__name__)
print("Feature Count:", len(bundle.feature_cols))
print("Feature Columns:", bundle.feature_cols)
print("Metadata:", bundle.metadata)
print("Imputer:", type(bundle.imputer).__name__)
print("Scaler:", type(bundle.scaler).__name__, bundle.scaler.feature_range)

