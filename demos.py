import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


from robustness.robustness_analysis import (
    robustness_experiment,
    robustness_ranking
)

from robustness.visualization import (
    plot_degradation,
    plot_cost
)

# ------------------ MODEL COMPARISON ------------------
def model_comparison(models, X_test, y_test, le):
    st.header("Model Comparison")

    metrics = []
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0)
        })

    df = pd.DataFrame(metrics)
    st.dataframe(df.style.format("{:.2f}"))

    st.subheader("Confusion Matrices")
    tabs = st.tabs(models.keys())

    for name, tab in zip(models.keys(), tabs):
        with tab:
            cm = confusion_matrix(y_test, models[name].predict(X_test))
            fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
            ax.matshow(cm, cmap="Blues", alpha=0.3)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)

            ax.set_xticklabels([""] + list(le.classes_), fontsize=8)
            ax.set_yticklabels([""] + list(le.classes_), fontsize=8)
            st.pyplot(fig)

# ------------------ TRICKY PACKETS ------------------
def tricky_packets(models, X_test, y_test):
    st.header("Tricky Packets Demo")
    n = st.slider("Number of tricky packets", 1, 10, 5)

    idx = random.sample(range(X_test.shape[0]), n)
    X_mod = X_test[idx] + np.random.normal(0, 0.5, X_test[idx].shape)
    y_true = y_test[idx]

    rows = []
    for i in range(n):
        row = {"True": y_true[i]}
        for name, model in models.items():
            row[name] = model.predict(X_mod[i].reshape(1, -1))[0]
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df)

# ------------------ NOISE STRESS TEST ------------------
def noise_stress(models, X_test, y_test):
    st.header("Noise Stress Test")

    noise_levels = [0, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for name, model in models.items():
        acc = []
        for n in noise_levels:
            X_noisy = X_test + np.random.normal(0, n, X_test.shape)
            acc.append(accuracy_score(y_test, model.predict(X_noisy)))
        results[name] = acc

    st.line_chart(pd.DataFrame(results, index=noise_levels))

# ------------------ MINORITY ATTACK FOCUS ------------------
def minority_attack_focus(models, X_test, y_test, attack_class=1):
    st.header("Minority Attack Detection")

    for name, model in models.items():
        preds = model.predict(X_test)
        idx = np.where(y_test == attack_class)[0]
        rate = (preds[idx] == attack_class).mean()
        st.write(f"**{name}** attack detection rate: `{rate:.2f}`")

# ------------------ ROBUSTNESS ANALYSIS ---------------------



def robustness_under_drift(models, X_test, y_test):
    st.header("Robustness Analysis Under Data Drift")

    st.markdown(
        """
        This experiment evaluates **how ensemble models behave when attack patterns change**.
        Instead of focusing on peak accuracy, we measure:
        - Performance degradation
        - False-negative risk
        - Cost-sensitive robustness
        """
    )

    # --- Drift controls ---
    max_drift = st.slider(
        "Maximum Feature Drift Intensity",
        min_value=0.0,
        max_value=0.6,
        value=0.4,
        step=0.05
    )

    drift_levels = np.arange(0.0, max_drift + 0.01, 0.05)

    # --- Run experiment ---
    with st.spinner("Running robustness experiment..."):
        results = robustness_experiment(
            models=models,
            X_test=X_test,
            y_test=y_test,
            drift_levels=drift_levels
        )

    # --- Accuracy degradation ---
    st.subheader("Performance Degradation")
    st.pyplot(plot_degradation(results))

    # --- Cost escalation ---
    st.subheader("Operational Cost Under Drift")
    st.pyplot(plot_cost(results))

    # --- Robustness ranking ---
    ranking_df = robustness_ranking(results)

    st.subheader("Robustness Ranking (Lower = Better)")
    st.dataframe(ranking_df)

    # --- Interpretation ---
    best_model = ranking_df.iloc[0]["Model"]

    st.success(
        f"**{best_model}** shows the highest robustness, "
        "exhibiting the smallest performance degradation and lowest operational risk under drift."
    )

    st.markdown(
        """
        ### Key Insight
        - Models with **high diversity** degrade more gracefully
        - Boosting-based methods achieve high accuracy on clean data but are **less stable**
        - Bagging-based ensembles are **more reliable for real-world intrusion detection**
        """
    )