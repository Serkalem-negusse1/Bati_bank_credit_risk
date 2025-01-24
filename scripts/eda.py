import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def summarize_data(df):
    """Prints summary statistics and data types."""
    print(df.info())
    print(df.describe())

def plot_distribution(df, column):
    """Plots the distribution of a numerical column."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_correlation_matrix(df):
    """Plots the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
