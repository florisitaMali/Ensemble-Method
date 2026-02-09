from sklearn.metrics import confusion_matrix

def cost_sensitive_score(y_true, y_pred, fn_cost=10, fp_cost=1):
    """
    Higher FN cost because missing attacks is dangerous
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn_cost * fn + fp_cost * fp
    return {
        "False Positives": fp,
        "False Negatives": fn,
        "Total Cost": total_cost
    }
