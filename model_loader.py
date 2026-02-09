import os
import pickle
from model_training import train_and_save_models

MODEL_NAMES = [
    "Weak Tree", "Bagging", "AdaBoost",
    "GradientBoosting", "Stacking", "Voting"
]

def load_models(X_train, y_train, model_dir="models"):
    models = {}
    missing = False

    for name in MODEL_NAMES:
        if not os.path.exists(f"{model_dir}/{name}.pkl"):
            missing = True
            break

    if missing:
        return train_and_save_models(X_train, y_train, model_dir)

    for name in MODEL_NAMES:
        with open(f"{model_dir}/{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)

    return models
