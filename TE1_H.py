# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:32:56 2023

@author: Bernardo
"""

#####################################Exercicio 1##########################################

import pandas as pd
import  os

# Carrega o arquivo csv em um dataframe
df1 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Life-Expectancy-Data-Updated.csv")

# Cria um novo dataframe com apenas a informação da região "South America"
df1_south_america = df1[df1['Region'] == 'South America']

# Grava o novo dataframe em um novo arquivo csv
df1_south_america.to_csv('Region_SA', index=False)

#linha de código para limpar variáveis e a consola do python
os.system('cls')


#####################################Exercicio 2##########################################

import pandas as pd
import  os
import matplotlib.pyplot as plt
import seaborn as sns
df2 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Region_SA.csv")

# Seleciona apenas os países desejados
paises = ['Bolivia', 'Colombia', 'Ecuador', 'Paraguay']
df_paises = df2[df2['Country'].isin(paises)]

# Cria um gráfico de linha para visualizar a evolução da mortalidade adulta nos países selecionados
fig, ax2 = plt.subplots()
for pais in paises:
    df_pais = df_paises[df_paises['Country'] == pais]
    sns.lineplot(df_pais['Year'], df_pais['Adult_mortality'], label=pais)

#Identificação dos eixos, título e legenda
ax2.set_xlabel('Ano')
ax2.set_ylabel('Mortes de adultos por 1000 habitantes')
ax2.set_title('Evolução da mortalidade adulta nos países selecionados')
ax2.legend()
#ax2.show()
os.system('cls')

#####################################Exercicio 3##########################################

import pandas as pd
import matplotlib.pyplot as plt
import  os
#carrega os dados em um DataFrame
df3 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Region_SA.csv")

#filtra os dados para conter apenas as informações desejadas
countries = ['Argentina', 'Brazil', 'Chile', 'Uruguay']
years = range(2000, 2016)
df_filtered = df3.loc[(df3['Country'].isin(countries)) & (df3['Year'].isin(years))]

#calcula a média do PIB per capita para cada país
gdp_means = []
for country in countries:
    gdp_mean = df_filtered.loc[df_filtered['Country'] == country]['GDP_per_capita'].mean()
    gdp_means.append(gdp_mean)

#Parte de criar o gráfico circular
labels = countries
sizes = gdp_means
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Média do PIB per capita nos países selecionados (2000-2015)')
plt.show()
os.system('cls')

#####################################Exercicio 4##########################################

import pandas as pd
import  os

#função para recolher os valores pedidos
def hiv_incidents(country):
    df4 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Region_SA.csv")
    df_country = df4[(df4["Country"] == country)]
    min_value = df_country["Incidents_HIV"].min()
    year = df_country.loc[df_country["Incidents_HIV"].idxmin(), "Year"]
    return f"Em {year}, o número de incidentes de HIV por 1000 habitantes dos 15 a 49 anos foi de {min_value}."

#Para usar a função temos que passar o país da South America como argumento
print(hiv_incidents("Argentina"))
os.system('cls')

#####################################Exercicio 5##########################################

import pandas as pd
import seaborn as sns
import  os

#Criamos 3 funções uma para cada questão do problema, assim é mais fácil 
#selecionar o que quer analisar 

#Função para analisar por pais
def grafico_dispersao_pais(pais):
    # Carrega o arquivo csv em um dataframe
    df5 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Life-Expectancy-Data-Updated.csv")

    # Seleciona apenas o país pretentido
    df_pais = df5[df5['Country'] == pais]

    # Cria um gráfico de dispersão com regressão linear
    sns.lmplot(x='Schooling', y='Life_expectancy', data=df_pais)

    plt.title(f'Escolaridade média vs. Esperança média de vida em {pais}')
    plt.xlabel('Anos médios de escolaridade')
    plt.ylabel('Esperança média de vida')
    plt.show()

#Função para analisar por regiao
def grafico_dispersao_regiao(regiao):
    
    # Carrega o arquivo csv em um dataframe
    df5 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Life-Expectancy-Data-Updated.csv")

    # Seleciona apenas a região pretendida
    df_regiao = df5[df5['Region'] == regiao]

    # Cria um gráfico de dispersão com regressão linear
    sns.lmplot(x='Schooling', y='Life_expectancy', data=df_regiao)

    plt.title(f'Escolaridade média vs. Esperança média de vida em {regiao}')
    plt.xlabel('Anos médios de escolaridade')
    plt.ylabel('Esperança média de vida')
    plt.show()


#Função para analisar no globo
def grafico_dispersao_globo():
    
    # Carrega o arquivo csv em um dataframe
    df5 = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Life-Expectancy-Data-Updated.csv")


    # Cria um gráfico de dispersão com regressão linear
    sns.lmplot(x='Schooling', y='Life_expectancy', data = df5)

    plt.title('Escolaridade média vs. Esperança média de vida no globo')
    plt.xlabel('Anos médios de escolaridade')
    plt.ylabel('Esperança média de vida')
    plt.show()

grafico_dispersao_pais("Spain")
grafico_dispersao_regiao("South America")
grafico_dispersao_globo()


#Explicação do declive da reta
#Quando a reta inclina-se para cima,
# isso significa que há uma correlação positiva entre as duas variáveis, 
#ou seja, quanto mais anos de escolaridade uma pessoa tiver,
# maior será sua esperança de vida. 
#Quando a reta inclina-se para baixo,
# isso significa que há uma correlação negativa entre as duas variáveis,
# ou seja, quanto mais anos de escolaridade uma pessoa tiver, 
# menor será sua esperança de vida. 
#Quando a reta é horizontal, isso significa que não há relação entre as duas variáveis.
os.system('cls')

#####################################Exercicio 6##########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv(r"C:\Users\Bernardo Reis\OneDrive\Ambiente de Trabalho\UTAD\3º Ano\2º Semestre\Introdução à ciência dos dados\TE1_H\Life-Expectancy-Data-Updated.csv")

sns.pairplot(df, x_vars=["Infant_deaths", "Under_five_deaths", "Adult_mortality", 
"Alcohol_consumption", "Hepatitis_B", "Measles", "BMI", "Polio", "Diphtheria", 
"Incidents_HIV", "GDP_per_capita", "Population_mln", "Thinness_ten_nineteen_years", 
"Thinness_five_nine_years", "Schooling", "Economy_status_Developed", 
"Economy_status_Developing"], y_vars=["Life_expectancy"])


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

X = df[["Adult_mortality", "BMI", "Hepatitis_B", "Polio", "Diphtheria", "GDP_per_capita", 
"Schooling", "Economy_status_Developing"]]
y = df["Life_expectancy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Real")
plt.ylabel("Predicton")
plt.show()

future_data = pd.DataFrame({"Adult_mortality": [200], "BMI": [25], "Hepatitis_B": [80], 
"Polio": [90], "Diphtheria": [90], "GDP_per_capita": [5000], "Schooling": [10], 
"Economy_status_Developing": [1]})
print("Previsão de esperança média de vida no futuro:", reg.predict(future_data))

os.system('cls')