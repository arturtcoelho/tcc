import pandas as pd

# Read the aggregated basic data
df = pd.read_csv('Agregados_por_bairros_basico_BR.csv', sep=';', encoding='latin-1')

# Filter on column x
df = df[df['NM_DIST'] == 'Curitiba']
df = df[['CD_BAIRRO', 'NM_BAIRRO', 'AREA_KM2', 'v0001', 'v0007']]

# Write filtered data to file
df.to_csv('Agregados_por_bairros_Curitiba.csv', index=False)
