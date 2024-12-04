import pandas as pd

# Load the dataset
data = pd.read_csv('results.csv')  # Replace with your CSV file path

# Calculate the required statistics for each metric
summary_stats = data[['mape', 'rmse', 'mae']].agg(['min', 'mean','max', 'std'])
summary_stats_transposed = summary_stats.T
# Print the summary statistics
print(summary_stats_transposed)