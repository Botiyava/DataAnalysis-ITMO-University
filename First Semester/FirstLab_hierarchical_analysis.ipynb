import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
%matplotlib inline

import os
os.chdir("/home/botiyava/learning/subjects/algo/lec3/eco228/")
df = pd.read_csv("Econom_Cities_data.csv", sep=";", index_col='City')
#Смотрим правильно ли открылся файл
df.head()
print(df)
#Смортим есть ли выбросы в нашей выборке
plt.show(df.boxplot(column=['Work']))
#Выбросы обнаружены, они равны -9999. 

#Таких значений всего 2 и мы можем исключить
#Эти города из выборки, т.к. 2 показателя из 3
#в них не несут какой-либо информации
q = df['Work'].quantile(0.01)
Z = df[df['Work'] > q]
print(Z)

#Стандартизируем данные, т.к. числа в 1 столбце намного больше чисел в остальных столбцах,
#но все столбцы одинакого значимы для нас.
from sklearn import preprocessing
norm = preprocessing.MinMaxScaler()
norm.fit(Z)
X = norm.transform(Z)

X = pd.DataFrame(X, index=Z.index, columns=Z.columns)

#Стандартизация прошла успешно
print(X)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
link = linkage(X, 'ward', 'euclidean')

print(type(link))
print(link.shape)

link

#Выводим дендрограмму, всё выглядит довольно хорошо
fig = plt.figure(figsize=(9,9))
dn = dendrogram(link, color_threshold = 0.9, labels=X.index.tolist())

dist = link[:, 2]
dist

#убедимся что мы выбрали правильное количество кластеров, используя каменистую осыпь
#Видно, что наибольшее изменение расстояние при 4 и 8 кластерах.
fig = plt.figure(figsize=(8,8))
dist_rev = dist[::-1]
idxs = range(1, len(dist) + 1)
plt.plot(idxs, dist_rev, marker='o')
plt.title('Distance between merged clusters')
plt.xlabel('Step')
plt.ylabel('Distance')

#Посмотрим какая дендрограмма будет при 8 кластерах
#По моему мнению 4 кластера выглядят лучше.
fig = plt.figure(figsize=(9,9))
dn = dendrogram(link, color_threshold = 0.49, labels=X.index.tolist())

X['cluster'] = fcluster(link, 4, criterion='maxclust')

print(X)

#1 кластер - работают средне, цены - маленькие, зарплаты - страны с уровнем жизни чуть меньше среднего 
#странно, что некоторые страны попали сюда, но по данным из выборки они находятся тут
#2 кластер - работают много,  цены - маленькие, зарплаты - маленькие : ,беднейшие страны из данной выборки
#3 кластер - работают - мало, цены - большие,   зарплаты - большие : богатые города с высоким уровнем жизни
#4 кластер - работают мало,   цены - средние,   зарплаты - выше среднего : обеспеченные города с хорошим уровнем
#жизни, присутствует много столиц
X.groupby('cluster').mean()

X.groupby('cluster').size()

#На всякий случай еще раз рассмотрим вариант с 8 кластерами, но уже подробнее
X['cluster'] = fcluster(link, 8, criterion='maxclust')

print(X)

#Кластеризация стала более точной, но различия между некоторыми кластерами слишком малы. 
#Лучшим решением будет оставить 4 кластера
X.groupby('cluster').mean()

X.groupby('cluster').size()

'''Выводы: В данной работе мною был проведён иерархический кластерный анализ, который показал,что в данной выборке
наиболее оптимально выглядят 4 кластера стран, каждый из которых разделен по критериям загруженности жителей, 
их зарплате и ценам в местных магазинах. Полученные кластеры можно охарактеризовать как кластер бедных стран, стран с нормальным
уровнем жизни,высоким и высочайшим(элитные города).
Количество стран в кластерах неравномерно, т.к. и в реальности экономическое развитие стран в мире неравномерно'''
