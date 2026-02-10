import streamlit as st
from data_utils import load_and_prepare_data
from model_loader import load_models
import demos

# configure the streamlit page settings
st.set_page_config(page_title="Ensemble Models", layout="wide")

# main title of the app
st.title("Weak vs Ensemble Models")

# load and preprocess the dataset
# this returns training/testing splits and label encoder
(X_train, X_test, y_train, y_test), le = load_and_prepare_data("cybersecurity_intrusion_data.csv", target_column="attack_detected")

# train and load all models (weak learners and ensemble models)
models = load_models(X_train, y_train)

# create a sidebar dropdown menu to select demo mode
mode = st.sidebar.selectbox(
    "Demo Mode",
    [
        "Model Comparison",
        "Tricky Packets",
        "Noise Stress Test",
        "Minority Attack Focus",
        "Robustness Under Drift" 
    ]
)

# run model comparison demo
if mode == "Model Comparison":
    demos.model_comparison(models, X_test, y_test, le)

# analyze difficult or borderline prediction samples
elif mode == "Tricky Packets":
    demos.tricky_packets(models, X_test, y_test)

# evaluate how models behave when noise is added
elif mode == "Noise Stress Test":
    demos.noise_stress(models, X_test, y_test)

# focus on performance for minority attack classes
elif mode == "Minority Attack Focus":
    demos.minority_attack_focus(models, X_test, y_test)

# test how models perform when data distribution changes (concept drift)
elif mode == "Robustness Under Drift":
    demos.robustness_under_drift(models, X_test, y_test)
