import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import geopandas as gpd


all = gpd.read_file('curitiba_bairros_completo.geojson')

selected_columns = [
    'areasVerdesM2',
    'popHabPHa',
    'popTotal',
    
    'popHomens', 
    'popBrancos',
    'porcentagem_brancos',
    'pop15a64',
    'porcentagem_adultos',
    'indiceEnvelhecimento',

    'rendaMediacRendimento',
    'rendaMedianacRendimento', 

    'area',
    'area_verde',
    'porcentagem_cobertura',

    'num_obitos_covid',
    'num_atendimentos_covid',
    'num_atendimentos_dengue',
    'num_atendimentos_saude_ja',
    'num_unidades_saude_bairro',
]

all = all[selected_columns]

all['porcentagem_cobertura'] = all['area_verde'] / all['area']

all['porcentagem_brancos'] = all['popBrancos'] / all['popTotal']
all['porcentagem_adultos'] = all['pop15a64'] / all['popTotal']

all['taxa_obitos_covid'] = all['num_obitos_covid'] / all['popTotal']
all['taxa_atendimentos_covid'] = all['num_atendimentos_covid'] / all['popTotal']
all['taxa_atendimentos_dengue'] = all['num_atendimentos_dengue'] / all['popTotal']
all['taxa_atendimentos_saude'] = all['num_atendimentos_saude_ja'] / all['popTotal']
all['taxa_unidades_saude'] = all['num_unidades_saude_bairro'] / all['popTotal']

selected_columns = [
    'popHabPHa',    
    'porcentagem_brancos',
    'porcentagem_adultos',
    'indiceEnvelhecimento',
    'rendaMediacRendimento',
    'rendaMedianacRendimento', 
    'porcentagem_cobertura',
    'taxa_obitos_covid',
    'taxa_atendimentos_covid',
    'taxa_atendimentos_dengue',
    'taxa_atendimentos_saude',
    'taxa_unidades_saude',
]
all = all[selected_columns]

### Correlation matrix
res = all.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(res, annot=True, cmap='coolwarm', fmt='.2f', 
            cbar=True, square=True, linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

# Save correlation matrix to CSV
# Fill NA values with column means
all = all.fillna(all.mean(numeric_only=True))
all.to_csv('dataTaxas.csv', index=False)
