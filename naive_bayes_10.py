# -*- coding: utf-8 -*-
"""Naive Bayes 10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KR4p-QYdLfs4OqbpNDKqLqJuVuxpIRln
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

df = pd.read_csv('Youtube02-KatyPerry.csv')
df1 = pd.read_csv('Youtube03-LMFAO.csv')
df2 = pd.read_csv('Youtube04-Eminem.csv')
df3 = pd.read_csv('Youtube05-Shakira.csv')
df4 = pd.concat([df, df1, df2, df3])
df4

for col in df.columns:
  labels, uniques = pd.factorize(df4[col])
  df4[col] = labels

y = df4['CLASS']
X = df4.drop(columns='CLASS')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Assuming multinomial distribution
nb_multi = MultinomialNB()
nb_multi.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
multi_preds = nb_multi.predict(X_test)
print(accuracy_score(y_test, multi_preds))
print(confusion_matrix(y_test, multi_preds))

# Assuming gaussian distribution
nb_gauss = GaussianNB()
nb_gauss.fit(X_train, y_train)

gauss_preds = nb_gauss.predict(X_test)
print(accuracy_score(y_test, gauss_preds))
print(confusion_matrix(y_test, gauss_preds))