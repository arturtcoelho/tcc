import pandas as pd

from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# Load your dataset
data = pd.read_csv('data.csv')

# data=(data-data.min())/(data.max()-data.min())

print(data)

# Separate features (X) and target variable (y)
X = data.drop('area_verde', axis=1)
X = X.drop('areasVerdesM2', axis=1)
X = X.drop('porcentagem_cobertura', axis=1)
y = data['porcentagem_cobertura']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=1234)
# Fit all regression models
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

print('done')

rmse = {}
mape = {}
r2sc = {}

for p in predictions:
    rmse[p] = mean_squared_error(y_test, predictions[p])
    mape[p] = mean_absolute_percentage_error(y_test, predictions[p])
    r2sc[p] = r2_score(y_test, predictions[p])

rmse = dict(sorted(rmse.items(), key=lambda item: item[1]))
mape = dict(sorted(mape.items(), key=lambda item: item[1]))
r2sc = dict(sorted(r2sc.items(), key=lambda item: item[1]))

for i in rmse:
    print(f'{i} : {rmse[i]}')
print()
for i in mape:
    print(f'{i} : {mape[i]}')
print()
for i in r2sc:
    print(f'{i} : {r2sc[i]}')

import matplotlib.pyplot as plt
import numpy as np

# Get top 5 models based on lowest MAPE values
top_5_models = list(mape.keys())[:5]

# Prepare data for plotting
models_names = top_5_models
rmse_values = [rmse[model] for model in top_5_models]
mape_values = [mape[model] for model in top_5_models]
r2_values = [r2sc[model] for model in top_5_models]

# Set up the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot RMSE
x = np.arange(len(models_names))
width = 0.25
ax1.bar(x - width, rmse_values, width, label='RMSE', color='b', alpha=0.7)
ax1.set_ylabel('RMSE', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Plot MAPE on the same axis as RMSE
ax2 = ax1.twinx()
ax2.bar(x, mape_values, width, label='MAPE', color='g', alpha=0.7)
ax2.set_ylabel('MAPE', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Plot R2 Score on a separate axis
ax3 = ax1.twinx()
ax3.bar(x + width, r2_values, width, label='R2 Score', color='r', alpha=0.7)
ax3.set_ylabel('R2 Score', color='r')
ax3.tick_params(axis='y', labelcolor='r')

# Adjust the R2 Score axis position
ax3.spines['right'].set_position(('axes', 1.2))

# Set x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(models_names, rotation=45, ha='right')

# Set title and legend
plt.title('Top 5 Models Performance Comparison')
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()