import os
import pickle
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

@st.cache_resource
def train_and_save_models(X_train, y_train, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    models = {
        "Weak Tree": DecisionTreeClassifier(max_depth=1, random_state=42),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42),
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42),
        "Stacking": StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=1)),
                ("svc", SVC(probability=True))
            ],
            final_estimator=LogisticRegression(),
            cv=5
        ),
        "Voting": VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=1)),
                ("bag", BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)),
                ("svc", SVC(probability=True))
            ],
            voting="soft"
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        with open(f"{model_dir}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    return models
