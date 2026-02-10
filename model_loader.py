# import os module for file and directory operations
import os

# import pickle to load saved models
import pickle

# import training function to retrain models if missing
from model_training import train_and_save_models


# define list of model names used as filenames
MODEL_NAMES = [
    "Weak Tree", "Bagging", "AdaBoost",
    "GradientBoosting", "Stacking", "Voting"
]


# define function to load trained models
# X_train and y_train are used only if retraining is required
# model_dir specifies folder where models are stored
def load_models(X_train, y_train, model_dir="models"):

    # initialize empty dictionary to store models
    models = {}

    # flag to track if any model file is missing
    missing = False

    # check if each model file exists
    for name in MODEL_NAMES:

        # construct file path and check existence
        if not os.path.exists(f"{model_dir}/{name}.pkl"):
            # mark as missing if any file does not exist
            missing = True
            # stop checking further
            break

    # if at least one model file is missing
    if missing:
        # retrain and save all models, then return them
        return train_and_save_models(X_train, y_train, model_dir)

    # if all model files exist
    for name in MODEL_NAMES:
        # open each model file in read-binary mode
        with open(f"{model_dir}/{name}.pkl", "rb") as f:
            # load model from file and store in dictionary using model name as key
            models[name] = pickle.load(f)

    # return dictionary containing all loaded models
    return models
