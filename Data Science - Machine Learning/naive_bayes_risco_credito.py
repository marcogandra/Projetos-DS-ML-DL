# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:45:18 2020

@author: olavo
"""

import pandas as pd
base = pd.read_csv('risco-credito.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

previsores[:,2] = labelencoder.fit_transform(previsores[:, 2])


from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

resultado = classificador.predict([[0, 1, 1, 2], [3, 0, 0, 0]])
resultado

classificador.classes_
classificador.class_count_
classificador.class_prior_