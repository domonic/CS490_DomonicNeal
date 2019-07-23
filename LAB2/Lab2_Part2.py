import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr
from pylab import rcParams

import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import TensorBoard

rcParams[('figure.figsize')] = 5, 4
sb.set_style('whitegrid')

cars_data = pd.read_csv("mtcars.csv")
cars_data.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs' , 'am','gear' ,'carb']

cars_data.head()

cars_sub = cars_data.iloc[:, [2, 4]].values
cars_data_names = ['cyl', 'hp']

y = cars_data.iloc[:, 9].values

sb.regplot(x='cyl', y='hp', data=cars_data, scatter=True)

cyl = cars_data['cyl']
hp = cars_data['hp']

spearmanr_coefficient, p_value = spearmanr(cyl, hp)

sb.countplot(x='am', data=cars_data, palette="hls")


X = scale(cars_sub)
LogReg = LogisticRegression()

LogReg.fit(X, y)

print(LogReg.score(X, y))

'''Predictors'''
y_pred = LogReg.predict(X)

print(classification_report(y, y_pred))

'''Create Model'''
model = Sequential()
model.add(Dense(32, input_shape=(32, 2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''Show the graph in Tensorboard'''
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, y, epochs=10, callbacks=[tensorboard], batch_size=32)

scores = model.evaluate(X, y, verbose=0)

model.save('cars_model.h5')

'''Output accuracy of the model'''
print('Test accuracy: %.2f%%' % (scores[1]*100))
print()
print('Test loss: ', scores[0])
print()
