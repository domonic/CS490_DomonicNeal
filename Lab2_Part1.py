<<<<<<< HEAD
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard


#PART 1A
'''Load default Fashion Dataset from Keras'''
fashion_mnist = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

'''Create Model'''
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''Show the graph in Tensorboard'''
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard])

scores = model.evaluate(X_test, y_test, verbose=0)

model.save('fashion_model.h5')


'''Output accuracy of the model'''
print('Test accuracy: %.2f%%' % (scores[1]*100))
print()
print('Test loss: ', scores[0])
print()


validation = [X_test[[i], ] for i in range(1000)]

'''Check prediction values'''
predictions = [np.argmax(model.predict(validate)) for validate in validation]
print("Prediction: ", predictions[0])

print()

'''Check actual values'''
actual = [np.argmax(x) for x in y_test[:1000]]

'''Compare prediction vs actual'''
for value in range(1000):
    print("Prediction: ", predictions[value])
    print("Actual value: ", actual[value])
=======
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard


#PART 1A
'''Load default Fashion Dataset from Keras'''
fashion_mnist = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

'''Create Model'''
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''Show the graph in Tensorboard'''
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard])

scores = model.evaluate(X_test, y_test, verbose=0)

model.save('fashion_model.h5')


'''Output accuracy of the model'''
print('Test accuracy: %.2f%%' % (scores[1]*100))
print()
print('Test loss: ', scores[0])
print()


validation = [X_test[[i], ] for i in range(1000)]

'''Check prediction values'''
predictions = [np.argmax(model.predict(validate)) for validate in validation]
print("Prediction: ", predictions[0])

print()

'''Check actual values'''
actual = [np.argmax(x) for x in y_test[:1000]]

'''Compare prediction vs actual'''
for value in range(1000):
    print("Prediction: ", predictions[value])
    print("Actual value: ", actual[value])
>>>>>>> d0a6132cf5e59ae6d1c75f24538aa712897cfe64
