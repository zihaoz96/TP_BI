#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC



# 3 preparations
x_label = "ProvPur"

df = pd.read_csv('KTFGHU14.csv', delimiter=';', header=[0])
# print(df)


# Nettoyage
dataNoMiss = df.iloc[0:,2:3]

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median= imp_median.fit(dataNoMiss)
df["Age"] = imp_median.transform(dataNoMiss)
SimpleImputer()

# Recodage
df["ProvPur"].replace({"No":0,"Yes":1}, inplace= True)
df["Region"].replace({"NE":0,"NW":1,"SE":2,"SW":3}, inplace= True)

nbr_yes = 0
for i in df["ProvPur"]:
    if i == 0:
        nbr_yes+=1
print(nbr_yes)
# print(df["Region"])



# # fig...
# print(df)
# .sum().reset_index()
# dfGroupByAge = df.groupby(x_label).sum().reset_index()
# x = dfGroupByAge[x_label]
# TotalPur = dfGroupByAge['TotalPur']


# plt.clf()
# fig = plt.figure(figsize=(16,8))
# ax = fig.add_subplot(111)
# ax.bar(x,TotalPur)

# # plt.plot(x, totalSpent, alpha=0.6)
# # plt.legend()
# plt.savefig("fig/"+x_label)

# Prétraitement

y_df = df["ProvPur"]
X_df = df.drop(["CityCode", "FirstPur", "LastPur", "NovelPur", "ChildPur", "YouthPur", "CookPur", "DiyPur", "ProvPur"], axis=1)


# Transform datas
scaler = StandardScaler()
X_df = scaler.fit_transform(X_df)

# Découpage
X_train = X_df[:3200] 
X_val = X_df[3200:3600]
X_test = X_df[3600:]

print(len(y_df))
y_train = y_df[:3200] 
y_val = y_df[3200:3600]
y_test = y_df[3600:]

# 4 Méthodes de classification
#######################################  SVM
svm = LinearSVC(
    max_iter = 1000,
)

multilabel_classifier = svm.fit(X_train, y_train)

# print("Score train: ", svm.score(X_train, y_train))
# print("Score val: ", svm.score(X_val, y_val))

res = multilabel_classifier.predict(X_test)

print("Method svc:")
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != res).sum()))


#######################################  KNN
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)

print("Method knn:")

res = nca_pipe.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_val.shape[0], (y_test != res).sum()))



#######################################  Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

print("Method gaussian naive bayes:")

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))











