import numpy as np
import pandas as pd

def apply_feature_drift(X, drift_strength=0.1, random_state=42):
    """
    Applies Gaussian noise to simulate feature drift
    """
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, drift_strength, X.shape)
    return X + noise


def apply_class_imbalance_drift(X, y, minority_ratio=0.05):
    """
    Simulates change in attack frequency (class imbalance drift)
    """
    df = pd.concat([X, y], axis=1)
    label_col = y.name

    minority = df[df[label_col] == 1]
    majority = df[df[label_col] == 0]

    minority_sample = minority.sample(
        frac=minority_ratio / (minority.shape[0] / df.shape[0]),
        replace=True,
        random_state=42
    )

    drifted_df = pd.concat([majority, minority_sample]).sample(frac=1)
    return drifted_df.drop(columns=[label_col]), drifted_df[label_col]
