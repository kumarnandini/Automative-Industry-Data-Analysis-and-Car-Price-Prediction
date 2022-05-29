# -*- coding: utf-8 -*-
# """
# Created on Thu May 26 21:55:53 2022

# @author: nandi
# """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

data = pd.read_csv("car_price.csv")

data['symboling'] = data['symboling'].astype('object')
import re
p = re.compile(r'\w+-?\w+')
datanames = data['CarName'].apply(lambda x: re.findall(p, x)[0])

data['data_company'] = data['CarName'].apply(lambda x: re.findall(p, x)[0])

# volkswagen
data.loc[(data['data_company'] == "vw") | 
         (data['data_company'] == "vokswagen")
         , 'data_company'] = 'volkswagen'

# porsche
data.loc[data['data_company'] == "porcshce", 'data_company'] = 'porsche'

# toyota
data.loc[data['data_company'] == "toyouta", 'data_company'] = 'toyota'

# nissan
data.loc[data['data_company'] == "Nissan", 'data_company'] = 'nissan'

# mazda
data.loc[data['data_company'] == "maxda", 'data_company'] = 'mazda'
data['data_company'].astype('category').value_counts()

predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)

features = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
prediction = model.predict(features)
print("Prediction: {}".format(prediction))





