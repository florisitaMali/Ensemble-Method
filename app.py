import streamlit as st
from data_utils import load_and_prepare_data
from model_loader import load_models
import demos

st.set_page_config(page_title="Ensemble Models", layout="wide")
st.title("Weak vs Ensemble Models")

(X_train, X_test, y_train, y_test), le = load_and_prepare_data(
    "cybersecurity_intrusion_data.csv",
    target_column="attack_detected"
)

models = load_models(X_train, y_train)

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

if mode == "Model Comparison":
    demos.model_comparison(models, X_test, y_test, le)

elif mode == "Tricky Packets":
    demos.tricky_packets(models, X_test, y_test)

elif mode == "Noise Stress Test":
    demos.noise_stress(models, X_test, y_test)

elif mode == "Minority Attack Focus":
    demos.minority_attack_focus(models, X_test, y_test)

elif mode == "Robustness Under Drift":
    demos.robustness_under_drift(models, X_test, y_test)  
