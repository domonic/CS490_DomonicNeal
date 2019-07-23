from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt
import tensorflow as tf
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

#PART 1
'''ADD ANOTHER LAYER'''
layer_one = Dense(128, activation='relu')(input_img)
layer_two = Dense(64, activation='relu')(layer_one)
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(layer_two)
layer_three = Dense(64, activation='relu')(encoded)
layer_four = Dense(128, activation='relu')(layer_three)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

#PART3
'''Visualize the input, noisy input and reconstructed representation'''
plt.imshow(x_test[2].reshape(28, 28), cmap='gray')
plt.title("Original")
plt.show()

#introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

plt.imshow(x_test_noisy[2].reshape(28, 28), cmap='gray')
plt.title("Noisy")
plt.show()

x_new = x_test[2][np.newaxis]

prediction = autoencoder.predict(x_new)
plt.imshow(prediction.reshape(28, 28), cmap='gray')
plt.title("Reconstructed")
plt.show()


#PART4
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.show()

