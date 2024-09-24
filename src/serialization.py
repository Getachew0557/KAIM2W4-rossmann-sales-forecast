import os
import joblib
from datetime import datetime
import warnings
import logging

# Warnings and logging
warnings.filterwarnings("ignore")

def save_model(model, model_name):
    models_dir = '../notebook/models'  # Use relative path for Kaggle or other environments
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"{models_dir}/{model_name}_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    logging.info(f"Model saved as {model_filename}")
