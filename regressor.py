import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import shap
import joblib

data = pd.read_csv('dataTaxas.csv')

# data=(data-data.min())/(data.max()-data.min())

X = data.drop('taxa_atendimentos_dengue', axis=1)
y = data['taxa_atendimentos_dengue']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

etr = ExtraTreesRegressor(random_state=42)

etr.fit(X_train, y_train)
y_pred = etr.predict(X_test)

print('rmse', mean_squared_error(y_pred, y_test))
print('mape', mean_absolute_percentage_error(y_pred, y_test))
print('r2sc', r2_score(y_pred, y_test))

joblib.dump(etr, "model.pkl") 

## SHAP values
explainer = shap.Explainer(etr)
shap_values = explainer(X)

shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values, max_display=50)
shap.plots.bar(shap_values)
