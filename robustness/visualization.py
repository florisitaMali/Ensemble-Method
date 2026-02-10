import pandas as pd
import matplotlib.pyplot as plt

def plot_degradation(results):
    fig, ax = plt.subplots(figsize=(7, 4)) 

    for model, records in results.items():
        df = pd.DataFrame(records)
        ax.plot(df["Drift"], df["Accuracy"], marker="o", label=model)

    ax.set_xlabel("Drift Intensity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance Degradation Under Feature Drift")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_cost(results):
    fig, ax = plt.subplots(figsize=(7, 4)) 

    for model, records in results.items():
        df = pd.DataFrame(records)
        ax.plot(df["Drift"], df["Total Cost"], marker="o", label=model)

    ax.set_xlabel("Drift Intensity")
    ax.set_ylabel("Operational Cost")
    ax.set_title("Cost Escalation Under Drift")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    return fig
