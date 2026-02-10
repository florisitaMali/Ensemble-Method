import os # import os module for directory and file handling
import pickle # import pickle module for saving trained models to files
import streamlit as st # import streamlit library for web app functionality
from sklearn.tree import DecisionTreeClassifier # import decision tree classifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier 

# import logistic regression model
from sklearn.linear_model import LogisticRegression

# import support vector classifier
from sklearn.svm import SVC


# cache_resource annotation tells streamlit to cache the function result
# it prevents retraining models every time the app reruns
@st.cache_resource
def train_and_save_models(X_train, y_train, model_dir="models"):

    # create model directory if it does not exist
    # exist_ok=True prevents error if folder already exists
    os.makedirs(model_dir, exist_ok=True)

    # define dictionary containing model names as keys and model objects as values
    models = {

        # weak decision tree with depth 1
        # random_state ensures reproducibility
        "Weak Tree": DecisionTreeClassifier(max_depth=1, random_state=42),

        # bagging ensemble using decision tree as base estimator
        # n_estimators=50 means 50 trees are trained
        "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42),

        # adaboost ensemble using weak decision tree
        # n_estimators=50 means 50 boosting iterations
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42),

        # gradient boosting with 100 boosting stages
        # learning_rate controls contribution of each tree
        # max_depth=1 creates weak learners
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42),

        # stacking classifier combines multiple base models
        "Stacking": StackingClassifier(

            # list of base estimators
            estimators=[
                # decision tree base learner
                ("dt", DecisionTreeClassifier(max_depth=1)),
                # support vector classifier with probability output enabled
                ("svc", SVC(probability=True))
            ],

            # final estimator combines outputs of base learners
            final_estimator=LogisticRegression(),
            # cv=5 performs 5-fold cross validation for stacking
            cv=5
        ),

        # voting classifier combines predictions from multiple models
        "Voting": VotingClassifier(

            # list of models used in voting
            estimators=[
                # decision tree model
                ("dt", DecisionTreeClassifier(max_depth=1)),
                # bagging model with 10 estimators
                ("bag", BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)),
                # support vector classifier with probability enabled
                ("svc", SVC(probability=True))
            ],

            # soft voting averages predicted probabilities
            voting="soft"
        )
    }

    # iterate through each model in dictionary
    for name, model in models.items():
        # train model using training data
        model.fit(X_train, y_train)

        # open file in write-binary mode to save model
        with open(f"{model_dir}/{name}.pkl", "wb") as f:
            # save trained model using pickle
            pickle.dump(model, f)

    # return dictionary of trained models
    return models
