
import pandas as pd
import numpy as np

df = pd.read_csv('predictionsNew.csv')

def calculate_mape(actual, predicted):
    # Avoid division by zero and compute the MAPE
    non_zero = actual != 0  # Filter out zero actual values to avoid division by zero
    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
    return mape



print(calculate_mape(df['taxa_atendimentos_covid'], df['preds']))