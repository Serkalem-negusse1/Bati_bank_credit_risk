import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def plot_distribution(data, column):
    """Plot distribution of a specific column."""
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_correlation_matrix(data):
    """Plot correlation matrix."""
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
