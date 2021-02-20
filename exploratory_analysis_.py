#!/usr/bin/env python
# coding: utf-8

# In[148]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

def show_sample(data):
    print('Aleatory Sample:\n')
    return print(data.sample())


def show_null_data(data):
    print('\nNull Data:\n')
    return print(data.isnull().sum())


def show_data_type(data):
    print('\nData Types:\n')
    return print(data.dtypes)


def show_absolute_frequency(data, column):
    print('\nAbsolute Frequency {} Feature\n'.format(column))
    freq = data[column].value_counts()
    return print(freq)


def show_outliers(data):
    print('\nIdentify Outliers\nBoxplot of each feature:\n')
    for i in data.describe().columns:
        sns.boxplot(data[i])
        plt.show()
    return None


def show_number_clusters(data, columns):
    print('\nFinding ideal number of clusters by Elbow Method:\n')
    pad = StandardScaler()
    data = data[columns].copy()
    data = pad.fit_transform(data)
    diss = []
    for i in np.arange(1,20):
        kmeans = KMeans(i)
        kmeans.fit(data)
        diss.append(kmeans.inertia_)
    plt.figure(figsize=[10,5])
    plt.plot(np.arange(1,20), diss)
    plt.xticks(np.arange(1,20))
    plt.show()
    return None


def data_clustering(data, columns, n_clusters):
    print('\nClustering Database by {}...\n'.format(columns))
    pad = StandardScaler()
    km = data[columns].copy()
    km = pad.fit_transform(km)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(km)
    print('Creating cluster feature...\n')
    data['cluster'] = kmeans.labels_
    print('Clusters Analysis:\n')
    centers = round(pd.DataFrame(pad.inverse_transform(kmeans.cluster_centers_), columns=columns),2)
    print(centers)
    print('\n')
    return data
    
    

def transform_data_type(data, column, dtype):
    print('\n\nData Type Transformation:\n')
    data[column] = data[column].astype(dtype)
    print('\n\nFeature {} is {} now.\n'.format(column, dtype))
    return data


def create_feature_decade(data):
    print('\nCreating a new feature:\n')
    data['DecadaConstrucao'] = data.AnoConstrucao
    for i in data.DecadaConstrucao.unique():
        if i >= 1940 and i < 1950:
            data['DecadaConstrucao'].replace(i, 40, regex=True, inplace=True)
        elif i >= 1950 and i < 1960:
            data['DecadaConstrucao'].replace(i, 50, regex=True, inplace=True)
        elif i >= 1960 and i < 1970:
            data['DecadaConstrucao'].replace(i, 60, regex=True, inplace=True)
        elif i >= 1970 and i < 1980:
            data['DecadaConstrucao'].replace(i, 70, regex=True, inplace=True)
        elif i >= 1980 and i < 1990:
            data['DecadaConstrucao'].replace(i, 80, regex=True, inplace=True)
        elif i >= 1990 and i < 2000:
            data['DecadaConstrucao'].replace(i, 90, regex=True, inplace=True)
        elif i >= 2000 and i < 2010:
            data['DecadaConstrucao'].replace(i, 2000, regex=True, inplace=True)
    data['DecadaConstrucao'] = data['DecadaConstrucao'].astype(str)
    data.sort_values('DecadaConstrucao', ascending=False, inplace=True)
    print('\nFeature Construction Decade Created.\n')
    print(data.sample())
    return data


def show_distributions(data):
    print('\nNumeric Features Distribuition:\n')
    data.hist(color = 'purple', figsize=(15,15))
    plt.show()
    return None


def show_stats(data):
    print('\nDescriptive Statistics:\n')
    return print(round(data.describe()))


def show_avg_value(data, column, value, n, hue = None):
    plt.figure(figsize=[12,5])
    sns.barplot(x = data[column].astype(int), y = data[value], hue = hue, data = data,
                palette = sns.color_palette('Blues', n))
    plt.title('{} X {}'.format(value, column), fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()
    return None
    

def show_correlations(data):
    plt.figure(figsize=[15, 5])
    sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1)
    plt.title('Correlations', fontsize=12)
    plt.show()
    return None


def save_cluster(data):
    print('Saving data cluster...')
    print('The biggest cluster: {}'.format(data.cluster.value_counts().index[0]))
    data = data[data['cluster'] == data.cluster.value_counts().index[0]].copy()
    data.drop(['DecadaConstrucao','cluster'], axis = 1, inplace = True)
    print('Saved successfully')
    return data.to_csv('cluster_for_predict.csv')
    

def data_collect( path ):
    # Extraction
    data = pd.read_csv(path, sep = ";").set_index("Id")
    return data


def data_analysis( data ):
    ## Extraction Analysis
    show_sample(data)
    show_null_data(data)
    show_data_type(data)
    show_absolute_frequency(data, column='Qualidade')
    show_outliers(data)
    show_number_clusters(data, columns=['Area','Preco'])
    return data


def data_transform(data):
    # Transformation
    data = transform_data_type(data, column = 'Qualidade', dtype = 'str')
    show_data_type(data)
    data = create_feature_decade(data)
    show_data_type(data)
    data = data_clustering(data, columns=['Area','Preco'], n_clusters=6)
    show_sample(data)
    data = transform_data_type(data, column='cluster', dtype = 'str')
    show_data_type(data)
    return data


def data_load(data):
    # Loading Results
    show_distributions(data)
    show_stats(data)
    show_avg_value(data, column='DecadaConstrucao', value='Preco', n = data['DecadaConstrucao'].nunique())
    show_avg_value(data, column='Comodos', value = 'Preco', hue = 'Zona', n = 2)
    show_avg_value(data, column='Qualidade', value='Preco', hue = 'QualidadeAquecimento', n = 4)
    show_correlations(data)
    show_absolute_frequency(data, column = 'cluster')
    # Saving the biggest cluster
    save_cluster(data)
    return None


if __name__ == '__main__':
    # ETL
    
    # Collect
    data_raw = data_collect('input_data.csv')
    
    # Analysis
    data_pre = data_analysis( data_raw )
    
    # Transform
    data_pro = data_transform( data_pre )
    
    # Load
    data_load( data_pro )


# Em média os imóveis possuem uma área de 8932 m², 6 comodos, 1 lareira, 2 garagens.
# 
# A maioria possui 1 banheiro, 3 quartos do tipo 1 e 4 quartos do tipo 2.
# 
# Preço médio de R$ 148.235,00, valor muito próximo da mediana de 147.000,00.
# 
# 
# 
# Zona RL não possui imóveis com 3 cômodos.
# 
# Zona RM não possui imóveis com 8 cômodos.
# 
# 
# 
# Zona RM tem preço médio entre 110.000,00 e 130.000,00, independente do número de cômodos.
# 
# Zona RL possui aumento significativo no preço conforme aumenta a quantidade de cômodos.
# 
# 
# 
# É possível determinar, por exemplo, que um imóvel de qualidade dos materiais = 7 e qualidade do aquecimento = Ex tem um preço médio de 180.000 enquanto um imóvel na  mesma qualidade de aquecimento, porém na categoria 4 de materiais tem preço médio de 120.000.
# 
# 
# 
# Area tem correlação positiva com todas as outras, exceto AnoConstrucao isso indica que a area do imovel tende a diminuir se o imóvel foi construído recentemente.
# 
# 
# Em relação a variável alvo (Preco), possui correlacao positiva com todas as outras, porém de forma mais forte com a quantidade de banheiros e de garagens do imóvel, ou seja, o preço tende a aumentar quando aumentamos o número de banheiros e garagens do imóvel.
