import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_percentage_error

data = pd.read_csv('test.csv')
# data2 = pd.read_csv('train.csv')

# data = pd.concat([data, data2], ignore_index=True)

# data=(data-data.min())/(data.max()-data.min())

X = data.drop(['Localidade',
                'taxa_atendimentos_covid'], axis=1)

y = data['taxa_atendimentos_covid']

model = joblib.load("modelFixed.pkl")
pred = model.predict(X)

data['preds'] = pred

data['erro_absoluto'] = data['preds'] - data['taxa_atendimentos_covid']

print(data)
data.to_csv('predictionsBoth.csv', index=False)

print(mean_absolute_percentage_error(y, pred))
