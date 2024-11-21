import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
# from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

import shap
import joblib

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.drop(['taxa_atendimentos_covid', 'Localidade'], axis=1)
y_train = train_data['taxa_atendimentos_covid']

X_test = test_data.drop(['taxa_atendimentos_covid', 'Localidade'], axis=1)
y_test = test_data['taxa_atendimentos_covid']


etr = ExtraTreesRegressor(random_state=42)

etr.fit(X_train, y_train)

y_pred = etr.predict(X_test)

metrics = {
    'MSE': mean_squared_error(y_pred, y_test),
    # 'RMSE': root_mean_squared_error(y_pred, y_test),
    'MAE': mean_absolute_error(y_pred, y_test),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred), 
    # 'R2': r2_score(y_pred, y_test),
}
print(metrics)

joblib.dump(etr, "modelFixed.pkl") 


data = pd.concat([X_train, X_test], ignore_index=True)

## SHAP values
explainer = shap.Explainer(etr)
shap_values = explainer(X_test)

shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[1])
shap.plots.beeswarm(shap_values, max_display=50)
shap.plots.bar(shap_values)
