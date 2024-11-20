import pandas as pd

df = pd.read_csv('raw/2024-10-23_Casos_Covid_19_-_Base_de_Dados.csv', sep=';')
                #  , encoding="ISO-8859-1")

print(df)

df = df[['BAIRRO', 'ENCERRAMENTO']]

# df = df[df['ENCERRAMENTO'].str.lower() == 'obito']

df_obitos = df[df['ENCERRAMENTO'].str.lower() == 'obito'].groupby('BAIRRO').size().reset_index(name='num_obitos_covid')
df_recuperados = df[df['ENCERRAMENTO'].str.lower() == 'recuperado'].groupby('BAIRRO').size().reset_index(name='num_recuperados_covid')
df_total = df.groupby('BAIRRO').size().reset_index(name='num_atendimentos')

# Merge the dataframes
df = df_obitos.merge(df_recuperados, on='BAIRRO', how='outer')
df = df.merge(df_total, on='BAIRRO', how='outer')

print(df)

df.to_csv('covid.csv')
