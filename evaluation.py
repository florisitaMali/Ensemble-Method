import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# used to predict the performance metrics
# y_true contains true labels
# y_pred contains predicted labels
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

# define function to compute attack detection rate
# y_true contains true labels
# y_pred contains predicted labels
# attack_class=1 defines which class represents attack
def attack_detection_rate(y_true, y_pred, attack_class=1):

    # find indices where true label equals attack class
    # np.where returns tuple so [0] extracts index array
    idx = np.where(y_true == attack_class)[0]

    # compare predictions at those indices with attack class
    # boolean array is produced and mean calculates proportion of correct detections
    return (y_pred[idx] == attack_class).mean()

