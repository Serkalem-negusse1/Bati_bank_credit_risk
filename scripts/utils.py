import os

def save_results(data, filepath):
    """Save results to a CSV file."""
    data.to_csv(filepath, index=False)

