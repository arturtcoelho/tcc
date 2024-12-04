import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np

import shap
import joblib

data = pd.read_csv('dataTaxas.csv')

# data=(data-data.min())/(data.max()-data.min())

# X = data.drop([ 'Localidade',
#                 'taxa_obitos_covid',
#                 'taxa_atendimentos_covid',
#                 'taxa_atendimentos_dengue',
#                 'taxa_atendimentos_saude', 
#                 'taxa_unidades_saude'], axis=1)

X = data.drop(['Localidade', 'taxa_atendimentos_covid'], axis=1)

y = data['taxa_atendimentos_covid']

from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=123)

param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000],       # Number of trees
    'learning_rate': [0.01, 0.1, 0.2],    # Step size shrinkage
    'max_depth': [3, 5, 7],               # Depth of each tree
    'subsample': [0.6, 0.8, 1.0],         # Fraction of samples used for training
    'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree
    'gamma': [0, 0.1, 0.2],               # Minimum loss reduction for split
    'reg_alpha': [0, 1, 10],              # L1 regularization
    'reg_lambda': [1, 10, 100]            # L2 regularization
}

etr = XGBRegressor(random_state=15)
grid_search = GridSearchCV(etr, param_grid, cv=3, scoring='neg_mean_absolute_percentage_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = root_mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

print(mape)

exit()

## SHAP values
explainer = shap.Explainer(etr)
shap_values = explainer(X)

shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values, max_display=50)
shap.plots.bar(shap_values)
