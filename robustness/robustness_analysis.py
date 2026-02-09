import numpy as np
from robustness.drift import apply_feature_drift
from robustness.cost import cost_sensitive_score
from sklearn.metrics import accuracy_score, f1_score

def robustness_experiment(models, X_test, y_test, drift_levels):
    """
    Runs models under increasing drift intensity
    """
    results = {}

    for name, model in models.items():
        results[name] = []

        for drift in drift_levels:
            X_drifted = apply_feature_drift(X_test, drift_strength=drift)
            y_pred = model.predict(X_drifted)

            metrics = {
                "Drift": drift,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
            }

            metrics.update(cost_sensitive_score(y_test, y_pred))
            results[name].append(metrics)

    return results


import pandas as pd

def robustness_ranking(results):
    """
    Rank models by degradation + cost
    """
    ranking = []

    for model, records in results.items():
        df = pd.DataFrame(records)

        acc_drop = df["Accuracy"].iloc[0] - df["Accuracy"].iloc[-1]
        avg_cost = df["Total Cost"].mean()

        score = acc_drop + 0.001 * avg_cost  # weighted robustness score

        ranking.append({
            "Model": model,
            "Accuracy Drop": acc_drop,
            "Avg Cost": avg_cost,
            "Robustness Score": score
        })

    ranking_df = pd.DataFrame(ranking).sort_values("Robustness Score")
    return ranking_df
