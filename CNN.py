# CNN model for spam dataset
import numpy as np
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten, Embedding
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# # fix random seed for reproducibility
seed = 7
np.random.seed(seed)

data = pd.read_csv('spam.csv', encoding='windows-1252')
# Keeping only the neccessary columns
data = data[['v2', 'v1']]

data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)

X = pad_sequences(X, maxlen=200)

embed_dim = 128

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Part 1 ICP11
try:
    model = load_model('new_model.h5')
except OSError:
    verbose, epochs, batch_size = 0, 10, 32
    # n_timesteps, n_features = X_train.shape[0], X_train.shape[1]
    model = Sequential()
    # model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(200, 1),
                     kernel_constraint=maxnorm(3)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    lrate = 0.01
    decay = lrate / epochs

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[tbCallBack])
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    model.save('new_model.h5')


test_cases = [X_test[[i], ] for i in range(4)]
predictions = [np.argmax(model.predict(case)) for case in test_cases]

actual_values = [np.argmax(x) for x in y_test[:4]]
for x in range(4):
    print("Prediction: ", predictions[x])
    print("Actual value: ", actual_values[x])

