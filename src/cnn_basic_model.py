import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, Activation, MaxPooling1D, Dropout
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model, Sequential


def cnn_preprocessor(tweet):
    """
    Take a tweet - which is just a simple text string - and remove or replace:
    1) Mentions i.e. @someone
    2) URLs i.e. https://www.twitter.com
    3) Repeated whitespace
    4) HTML entities
    5) punctuation and numbers
    :param tweet: text string
    :return preprocessed_tweet: without mentions, URLs and extra whitespace
    """
    re_mention = '@[\w\-]+'
    re_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    re_whitespace = '\s+'
    re_html_entity = '&[^\s]*;' # discovered that Davidson et al. was not utf-8 encoded
    re_punct = '[^\w\s]|\d|\_'
    preproc_tweet = re.sub(re_html_entity, ' ', tweet)
    preproc_tweet = re.sub(re_mention, ' ', preproc_tweet)
    preproc_tweet = re.sub(re_url, ' ', preproc_tweet)
    preproc_tweet = re.sub(re_punct, ' ', preproc_tweet)
    preproc_tweet = re.sub(re_whitespace, ' ', preproc_tweet)
    return preproc_tweet


def cnn_tokenizer(tweet):
    """
    Take a tweet - a simple text string - and set characters to
    lowercase and stem the tweets.
    :param tweet: a text string
    :return tokenized_tweet: all lowercase, no punctuation and stemmed
    """
    tweet_lower = tweet.lower()
    stemmer = nltk.PorterStemmer()
    tokenized_tweet = ' '.join([stemmer.stem(w) for w in tweet_lower.split()])
    return tokenized_tweet


def cnn_stopwords(tweet):
    """Remove stopwords from a tweet"""
    stop_words = nltk.corpus.stopwords.words('english')
    other_words = ['rt', 'ff', 'tbt', 'ftw']  # may add more later
    stop_words.extend(other_words)
    stopped_tweet = ' '.join([w for w in tweet.split(' ') if w not in stop_words])
    return stopped_tweet


def train_model(data):
    """Train the CNN model"""

    # get and preprocess the data
    tweet_df = pd.read_csv(data)
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: cnn_stopwords(x))
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: cnn_tokenizer(x))
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: cnn_preprocessor(x))
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
    embedding_size = len(character_encoding) + 1  # includes padding

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

    # embeddings and inputs?



    # build the model
    model = Sequential()

    # need to figure out input layer
    model.add(Conv1D(256, 7, activation='relu', input_shape=(seq_length, 100)))
    model.add(MaxPooling1D(3))



    model.add(Conv1D(256, 7, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return


if __name__ == '__main__':
    train_model("data/private/td_zw_labeled_data.csv")
