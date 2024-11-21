import pandas as pd
import joblib

data = pd.read_csv('dataTaxas.csv')

# data=(data-data.min())/(data.max()-data.min())

X = data.drop(['Localidade',
                'taxa_obitos_covid',
                'taxa_atendimentos_covid',
                'taxa_atendimentos_dengue',
                'taxa_atendimentos_saude',
                'taxa_unidades_saude'], axis=1)

y = data['taxa_atendimentos_covid']

model = joblib.load("model.pkl")
pred = model.predict(X)

data['preds'] = pred
print(data)
data.to_csv('predictionsNew.csv', index=False)