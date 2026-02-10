import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# import robustness experiment logic
from robustness.robustness_analysis import robustness_experiment, robustness_ranking

# import visualization utilities for robustness plots
from robustness.visualization import plot_degradation, plot_cost

# ------------------ MODEL COMPARISON ------------------

def model_comparison(models, X_test, y_test, le):
    # display section title
    st.header("Model Comparison")

    # store evaluation metrics for each model
    metrics = []

    # evaluate each model
    for name, model in models.items():
        preds = model.predict(X_test)

        # compute standard classification metrics
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0)
        })

    # convert results to dataframe
    df = pd.DataFrame(metrics)

    # display formatted metric table
    st.dataframe(df.round(2))

    # show confusion matrices for each model
    st.subheader("Confusion Matrices")

    # create one tab per model
    tabs = st.tabs(models.keys())
    for name, tab in zip(models.keys(), tabs):
        with tab:
            st.write(f"**{name} Confusion Matrix**")
            cm = confusion_matrix(y_test, models[name].predict(X_test))

            # create small heatmap
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=le.classes_, yticklabels=le.classes_,
                        annot_kws={"size": 8})
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)



# ------------------ TRICKY PACKETS ------------------
def tricky_packets(models, X_test, y_test):
    # display section title
    st.header("Tricky Packets Demo")

    st.markdown(
        """
        Purpose:  
        This demo shows how models behave on borderline or tricky packets—inputs that are slightly modified to be hard to classify.  
        It highlights which models are robust to subtle changes and which make mistakes on uncertain samples.
        """
    )

    # allow user to select number of packets
    n = st.slider("Number of tricky packets", 1, 10, 5)

    # randomly select samples
    idx = random.sample(range(X_test.shape[0]), n)

    # add gaussian noise to simulate borderline or tricky cases
    X_mod = X_test[idx] + np.random.normal(0, 0.5, X_test[idx].shape)

    # true labels of selected samples
    y_true = y_test[idx]

    rows = []

    # compare predictions of all models on modified samples
    for i in range(n):
        row = {"True": y_true[i]}
        for name, model in models.items():
            row[name] = model.predict(X_mod[i].reshape(1, -1))[0]
        rows.append(row)

    # create dataframe
    df = pd.DataFrame(rows)

    # function to color cells
    def color_preds(val, true_val):
        if val == true_val:
            return 'background-color: #b6fcd5'  # green for correct
        else:
            return 'background-color: #fcb6b6'  # red for incorrect

    # apply coloring
    styled_df = df.style.apply(lambda row: [color_preds(v, row['True']) for v in row], axis=1)

    # display comparison table
    st.dataframe(styled_df)



# ------------------ NOISE STRESS TEST ------------------

def noise_stress(models, X_test, y_test):
    # display section title
    st.header("Noise Stress Test")

    st.markdown(
        """
        Purpose:  
        Real-world inputs can be noisy or slightly corrupted.  
        This test evaluates how each model's accuracy degrades as noise intensity increases.  
        Models that maintain higher accuracy under noise are considered more robust to data imperfections.
        """
    )

    # predefined noise intensity levels
    noise_levels = [0, 0.1, 0.2, 0.3, 0.5]
    results = {}

    # evaluate each model under increasing noise
    for name, model in models.items():
        acc = []
        for n in noise_levels:
            # add gaussian noise to entire test set
            X_noisy = X_test + np.random.normal(0, n, X_test.shape)

            # compute accuracy under noisy conditions
            acc.append(accuracy_score(y_test, model.predict(X_noisy)))

        results[name] = acc

    # plot accuracy degradation curve
    st.line_chart(pd.DataFrame(results, index=noise_levels))


# ------------------ MINORITY ATTACK FOCUS ------------------

def minority_attack_focus(models, X_test, y_test, attack_class=1):
    # display section title
    st.header("Minority Attack Detection")

    st.markdown(
        """
        Purpose:  
        Real-world inputs can be noisy or slightly corrupted.  
        This test evaluates how each model's accuracy degrades as noise intensity increases.  
        Models that maintain higher accuracy under noise are considered more robust to data imperfections.
        """
    )

    # evaluate detection rate for minority attack class
    for name, model in models.items():
        preds = model.predict(X_test)

        # find indices of minority class
        idx = np.where(y_test == attack_class)[0]

        # compute detection rate for that class
        rate = (preds[idx] == attack_class).mean()

        # display detection performance
        st.write(f"{name} attack detection rate: `{rate:.2f}`")


# ------------------ ROBUSTNESS ANALYSIS ------------------

def robustness_under_drift(models, X_test, y_test):
    # display section title
    st.header("Robustness Analysis Under Data Drift")

    # explain purpose of experiment
    st.markdown(
        """
        This experiment evaluates how ensemble models behave when attack patterns change.
        Instead of focusing on peak accuracy, we measure:
        - Performance degradation
        - False-negative risk
        - Cost-sensitive robustness
        """
    )

    # slider to control maximum drift intensity
    max_drift = st.slider(
        "Maximum Feature Drift Intensity",
        min_value=0.0,
        max_value=0.6,
        value=0.4,
        step=0.05
    )

    # generate drift levels
    drift_levels = np.arange(0.0, max_drift + 0.01, 0.05)

    # run robustness experiment
    with st.spinner("Running robustness experiment..."):
        results = robustness_experiment(models=models, X_test=X_test, y_test=y_test, drift_levels=drift_levels)

    # explain performance degradation
    st.subheader("Performance Degradation")
    st.markdown(
        """
        **Why this matters:**  
        Models trained on historical data may face degraded performance when input features shift (data drift).  
        This plot shows how the accuracy or detection rate changes as drift intensity increases, highlighting models that maintain stability versus those that fail quickly.
        """
    )
    st.pyplot(plot_degradation(results))

    # explain operational cost
    st.subheader("Operational Cost Under Drift")
    st.markdown(
        """
        **Why this matters:**  
        Beyond accuracy, misclassifications have real-world consequences.  
        False negatives (missed detections) can be costly in intrusion detection.  
        This graph estimates the operational cost under drift, helping identify models that minimize risk even under challenging conditions.
        """
    )
    st.pyplot(plot_cost(results))

    # explain robustness ranking
    st.subheader("Robustness Ranking (Lower = Better)")
    st.markdown(
        """
        **Why this matters:**  
        Combining multiple metrics into a robustness ranking helps compare models holistically.  
        A lower ranking means the model performs consistently well across drift levels, maintaining low error and low operational cost.
        """
    )
    ranking_df = robustness_ranking(results)
    st.dataframe(ranking_df)

    # identify most robust model
    best_model = ranking_df.iloc[0]["Model"]

    # display final conclusion
    st.success(
        f"{best_model} shows the highest robustness, exhibiting the smallest performance degradation and lowest operational risk under drift."
    )

    # provide conceptual insight
    st.markdown(
        """
        ### Key Insight
        - Models with high diversity degrade more gracefully (less correlated errors)
        - Boosting-based methods achieve high accuracy on clean data but are less stable under drift
        - Bagging-based ensembles are more reliable for real-world intrusion detection because they are less sensitive to small changes
        """
    )
