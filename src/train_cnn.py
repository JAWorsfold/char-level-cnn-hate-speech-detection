import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, Activation, MaxPooling1D, Dropout
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model, Sequential
from src.preprocess_utils import PreProcessUtils


def train_model(data):
    """Train the CNN model"""
    pp = PreProcessUtils()

    # get and pre-process the data
    tweet_df = pd.read_csv(data)
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: pp.normalise(pp.remove_noise(x)))
    y = tweet_df['class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(tweet_df['tweet'], y, random_state=33, test_size=0.1)

    # create character dictionary as vocabulary and create tokenizer
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(X_train)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    character_encoding = {}
    for i, c in enumerate(alphabet):
        character_encoding[c] = i + 1
    tokenizer.word_index = character_encoding.copy()

    # cnn parameters
    max_len = 140
    vocab_size = len(character_encoding)
    embedding_size = len(character_encoding)

    # create one-hot-encodings
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test  = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
    X_test  = pad_sequences(X_test, maxlen=max_len, padding='post')

    X_train = np.array(X_train, dtype=np.float32)
    X_test  = np.array(X_test, dtype=np.float32)

    embeddings = list()
    embeddings.append(np.zeros(vocab_size))
    for c, i in tokenizer.word_index.items():
        one_hot = np.zeros(vocab_size)
        one_hot[i - 1] = 1
        embeddings.append(one_hot)
    embeddings = np.array(embeddings)

    # build the model
    model = Sequential()

    # input layer
    # model.add(Input(shape=(max_len,), dtype=np.int64))
    model.add(Embedding(vocab_size + 1, embedding_size, input_length=max_len, weights=[embeddings]))

    # convolution and pooling layers
    model.add(Conv1D(256, 7, activation='relu', input_shape=(max_len,), input_dtype=np.int64))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(256, 7, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    # train the model
    history = model.fit(X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=100,
                epochs=10,
                verbose=2)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
    print("Training Accuracy: {:.5f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    print("Testing Accuracy: {:.5f}".format(accuracy))

    return


if __name__ == '__main__':
    train_model("data/private/td_zw_labeled_data.csv")
