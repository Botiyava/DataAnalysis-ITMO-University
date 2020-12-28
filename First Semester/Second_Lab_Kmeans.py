import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
%matplotlib inline

import os
os.chdir("/home/botiyava/learning/subjects/algo/lec3/eco228/")

df = pd.read_csv("Econom_Cities_data.csv", sep=";", index_col='City')
df.head()

#Смортим есть ли выбросы в нашей выборке
plt.show(df.boxplot(column=['Work']))

#Выбросы обнаружены, они равны -9999. 
#Таких значений всего 2 и мы можем исключить
#Эти города из выборки, т.к. 2 показателя из 3
#в них не несут какой-либо информации
q = df['Work'].quantile(0.01)
Z = df[df['Work'] > q]
Z.reset_index(drop=True)

print(Z)

#Стандартизируем данные, т.к. числа в 1 столбце намного больше чисел в остальных столбцах,
#но все столбцы одинакого значимы для нас.
from sklearn import preprocessing
norm = preprocessing.MinMaxScaler()
norm.fit(Z)
X = norm.transform(Z)

X = pd.DataFrame(X, index=Z.index, columns=Z.columns)

print(X)

from sklearn.cluster import KMeans

# Выбираем число кластеров. От 2 до 12
K = range(2, 13)

# Строим 10 моделей с разным числом кластеров
# Не рационально, зачем сохранять модель, достаточно хранить только model.inertia_
models = [KMeans(n_clusters=k, random_state=42, n_init=100, verbose=0).fit(X) for k in K]

#  Качество кластеризации содержится в model.inertia_
dist = [model.inertia_ for model in models]

# Строим график каменистая осыпь
plt.plot(K, dist, marker='o')
# Добавляем на график текст
plt.xlabel('k')
plt.ylabel(' сумма расстояний')
plt.title('Каменистая осыпь для определения числа кластеров')
plt.show()
#больше всего расстояние меняется при 4 и 5 кластерах, на всякий случай можно захватить еще и вариант с 6 кластерами

#Смотрим на 4 кластера
#Хотелось бы, чтобы кластер с индексом 1 тоже распался и выровнял картину распределения элементов по кластерам
model = KMeans(n_clusters=4, random_state=42, max_iter=300, n_init=10, verbose=0)
model.fit(X)
X['cluster'] = model.labels_
X.groupby('cluster').mean()

X['cluster'].sort_values()

#Смотрим на 5 кластеров
#как мы видим кластер с индексом 1 по прежнему не распался, значит там действительно очень похожие элементы
#и не стоит его принудительно раскалывать
model = KMeans(n_clusters=5, random_state=42, max_iter=300, n_init=10, verbose=0)
model.fit(X)
X['cluster'] = model.labels_
X.groupby('cluster').mean()

#кластер с индексом 3 при четырех кластерах разбился на кластеры с индексами 3 и 4 в варианте с 5 кластерами
#Нужно посмотреть, насколько это оправдано
X.groupby('cluster').size()

X['cluster'].sort_values()

#Смотрим на 6 кластеров, чтобы удостовериться, что это только сделает хуже. 
#Так и есть, теперь кластеров очень много и все они малы, а отличаются на незначительные значения.
model = KMeans(n_clusters=6, random_state=42, max_iter=300, n_init=10, verbose=0)
model.fit(X)
X['cluster'] = model.labels_
X.groupby('cluster').mean()

X.groupby('cluster').size()

#Возьмём случайные города из 3 и 4 кластера. Как мы видим они почти не различаются. По крайней мере намного меньше,
#чем если сравнитт их с городами из других кластеров. Поэтому мой вердикт - в данной выборке оптимально выделить 4 кластера
print("Buenos_Aires:")
print("Work: ",X['Work'][5])
print("Price: ",X['Price'][5])
print("Salary: ",X['Salary'][5])
print("\nSingpore:")
print("Work: ",X['Work'][37])
print("Price: ",X['Price'][37])
print("Salary: ",X['Salary'][37])

