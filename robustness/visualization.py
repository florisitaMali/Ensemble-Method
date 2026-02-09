import pandas as pd
import matplotlib.pyplot as plt

def plot_degradation(results):
    plt.figure()
    for model, records in results.items():
        df = pd.DataFrame(records)
        plt.plot(df["Drift"], df["Accuracy"], label=model)

    plt.xlabel("Drift Intensity")
    plt.ylabel("Accuracy")
    plt.title("Performance Degradation Under Feature Drift")
    plt.legend()
    plt.grid(True)
    return plt


def plot_cost(results):
    plt.figure()
    for model, records in results.items():
        df = pd.DataFrame(records)
        plt.plot(df["Drift"], df["Total Cost"], label=model)

    plt.xlabel("Drift Intensity")
    plt.ylabel("Operational Cost")
    plt.title("Cost Escalation Under Drift")
    plt.legend()
    plt.grid(True)
    return plt
