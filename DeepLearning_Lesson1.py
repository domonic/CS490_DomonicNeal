import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical


dataset = pd.read_csv("diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8], test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential()
my_first_nn.add(Dense(20, input_dim=8, activation='relu'))

'''add another more dense layer'''
my_first_nn.add(Dense(30, input_dim=8, activation='relu'))

my_first_nn.add(Dense(1, activation='sigmoid'))
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0, initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))

print()
print()

'''Read from CSV file'''
breast_dataset = pd.read_csv("Breas Cancer.csv")

'''Gather features from CSV'''
breast_dataset_x = breast_dataset.iloc[:, 2:32]


'''Target from CSV'''
breast_dataset_y = breast_dataset.iloc[:, 1]

'''Change the target from categorical to numerical'''
breast_dataset_y = to_categorical(breast_dataset_y.replace({'B': 0, 'M': 1}))


breast_dataset_x, x_test, breast_dataset_y, y_test = train_test_split(breast_dataset_x, breast_dataset_y, test_size=0.25, random_state=87)

model = Sequential()
model.add(Dense(20, input_dim=30, activation="relu"))
model.add(Dense(2, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_fitted = model.fit(breast_dataset_x, breast_dataset_y, epochs=100, verbose=0, initial_epoch=0)
print(model.summary())
print(model.evaluate(breast_dataset_x, breast_dataset_y, verbose=0))
















