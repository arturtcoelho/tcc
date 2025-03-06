import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
# from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np

import shap
import joblib

import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_csv('dataTaxas.csv')

X = data.drop(['Localidade', 'taxa_atendimentos_covid', 'taxa_obitos_covid'], axis=1)
y = data['taxa_atendimentos_covid']

mape = []
MSE = []
# RMSE = []
MAE = []
models = []
for i in range(15):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=i*123)
    model = ExtraTreesRegressor(max_depth=None, 
                       min_samples_leaf=10, 
                       min_samples_split=2,
                       n_estimators=500,
                        random_state=i*15)
    
    # model = RandomForestRegressor(random_state=i*15)
    # RandomForestRegressor = 0.158

    # model = LGBMRegressor(random_state=i*15)
    # LGBMRegressor = 0.158

    # model = XGBRegressor(random_state=i*7)
    # XGBRegressor = 0.175


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mape += [mean_absolute_percentage_error(y_test, y_pred)]
    MSE += [mean_squared_error(y_test, y_pred)]
    # RMSE += [root_mean_squared_error(y_test, y_pred)]
    MAE += [mean_absolute_error(y_test, y_pred)]

    models += [model]

avg_mape = sum(mape)/len(mape)
avg_mape_index = mape.index(min(mape, key=lambda x:abs(x-avg_mape)))
closest_model = models[avg_mape_index]

joblib.dump(closest_model, "model.pkl")

# print('mape, rmse, mae')
# for i in range(len(mape)):
#     print(f"{mape[i]}, {RMSE[i]}, {MAE[i]}")

print(mape)

print('mape' , sum(mape)/len(mape))
print('mse' , sum(MSE)/len(MSE))
# print('rmse' , sum(RMSE)/len(RMSE))
print('mae' , sum(MAE)/len(MAE))

# exit()

# SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

for x in X:
    shap.plots.scatter(shap_values[:, x], color=shap_values)
# for i in range(len(X)):
#     shap.plots.waterfall(shap_values[i])
shap.plots.beeswarm(shap_values, max_display=50)
shap.plots.bar(shap_values)
