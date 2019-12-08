import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
print(df.head())
print(df.shape)
##Univariant Analysis
print(df['species'].value_counts())
df_setosa=df.loc[df['species']=='setosa']
#print(df_setosa)
df_virginica=df.loc[df['species']=='virginica']
df_versicolor =df.loc[df['species']=='versicolor']
print(df_versicolor.head())
plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('sepal length')
plt.show()

## Bivariate Analysis
sns.FacetGrid(df,hue='species',size=5).map(plt.scatter,'petal_length','sepal_length').add_legend()
plt.show()

## Multivariate Analysis
sns.pairplot(df,hue='species',size=3)
plt.show()
