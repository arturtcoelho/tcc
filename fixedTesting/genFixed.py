import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('dataTaxas.csv')

X = data.drop([
    'taxa_obitos_covid',
    'taxa_atendimentos_covid', 
    'taxa_atendimentos_dengue',
    'taxa_atendimentos_saude',
    'taxa_unidades_saude'], axis=1)

y = data['taxa_atendimentos_covid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
