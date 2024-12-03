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

# print(X)

mape = []
MSE = []
RMSE = []
MAE = []
for i in range(15):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=i*123)
    etr = ExtraTreesRegressor(max_depth=None, 
                       min_samples_leaf=10, 
                       min_samples_split=2,
                       n_estimators=500,
                        random_state=i*15)

    etr.fit(X_train, y_train)
    y_pred = etr.predict(X_test)

    #     # Perform k-fold cross-validation
    # k=5
    # kf = KFold(n_splits=k, shuffle=True, random_state=i*1234)
    # scores = cross_val_score(etr, X, y, cv=kf, scoring='neg_mean_absolute_percentage_error')

    # # Convert negative MSE to positive RMSE for interpretability
    # rmse_scores = np.sqrt(-scores)

    # print(f"mape for each fold: {rmse_scores}")
    # print(f"Average mape: {rmse_scores.mean():.4f}")

    mape += [mean_absolute_percentage_error(y_test, y_pred)]
    MSE += [mean_squared_error(y_test, y_pred)]
    RMSE += [root_mean_squared_error(y_test, y_pred)]
    MAE += [mean_absolute_error(y_test, y_pred)]

print('mape' , sum(mape)/len(mape))
print('mse' , sum(MSE)/len(MSE))
print('rmse' , sum(RMSE)/len(RMSE))
print('mae' , sum(MAE)/len(MAE))

joblib.dump(etr, "model.pkl") 
exit()

## SHAP values
explainer = shap.Explainer(etr)
shap_values = explainer(X)

shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values, max_display=50)
shap.plots.bar(shap_values)
