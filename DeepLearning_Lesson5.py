import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=2)
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("Score Sentiment:", score)
print("Accuracy Sentiment: ", acc)
print(model.metrics_names)

''' PART1 Save the model and use the saved model to predict on new text data'''
model.save("./modelAnalysis.h5")

data = pd.Series("A lot of good things are happening. We are respected again throughout the world, and that's a great thing")

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data)
X = tokenizer.texts_to_sequences(data)

X = pad_sequences(X, maxlen=28)

print("Prediction: ", np.argmax(model.predict(X)))
print("Actual: ", np.argmax(Y_test[0]))
print()
print()

'''PART2 Apply GridSearchCV on the source code provided in the class'''
model = KerasClassifier(build_fn=createmodel, verbose=1)
batch_size = [10, 20]
epochs = [1, 2]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
print()
print("GridSearchCV")
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print()
print()

'''PART 3 Apply the code on spam data set available in the sourcecode (text classification on the spam.csv data set)'''
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
# Keeping only the neccessary columns
data = data[['v2','v1']]

data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=2)
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("Score Spam:", score)
print("Accuracy Spam: ", acc)
print(model.metrics_names)