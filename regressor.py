import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

import shap
import joblib

data = pd.read_csv('dataTaxas.csv')

# data=(data-data.min())/(data.max()-data.min())

X = data.drop([ 'Localidade',
                'taxa_obitos_covid',
                'taxa_atendimentos_covid',
                'taxa_atendimentos_dengue',
                'taxa_atendimentos_saude',
                'taxa_unidades_saude'], axis=1)

y = data['taxa_atendimentos_covid']

print(X)

m = []
for i in range(15):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=i)
    etr = ExtraTreesRegressor(random_state=i)

    etr.fit(X_train, y_train)
    y_pred = etr.predict(X_test)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred), 
        # 'R2': r2_score(y_pred, y_test),
    }
    m += [metrics]

# print(m)
for n in m:
    for i in n:
        print(n[i], end='')
        print(',', end='')
    print()

joblib.dump(etr, "model.pkl") 
exit()

## SHAP values
explainer = shap.Explainer(etr)
shap_values = explainer(X)

shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values, max_display=50)
shap.plots.bar(shap_values)
