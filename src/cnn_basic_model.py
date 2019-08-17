import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, Activation, MaxPooling1D
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model

# cnn parameters:
VOCAB_SIZE = 70
MAX_LEN = 140


def cnn_preprocessor(tweet):
    """
    Take a tweet - which is just a simple text string - and remove or replace:
    1) Mentions i.e. @someone
    2) URLs i.e. https://www.twitter.com
    3) Repeated whitespace
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
    Take a tweet - a simple text string - and remove punctuation, set characters to
    lowercase and stem the tweets.
    :param tweet: a text string
    :return tokenized_tweet: all lowercase, no punctuation and stemmed
    """
    tweet_lower = tweet.lower()
    stemmer = nltk.PorterStemmer()
    tokenized_tweet = ' '.join([stemmer.stem(w) for w in tweet_lower.split()])
    return tokenized_tweet


def cnn_stopwords(tweet):

    stop_words = nltk.corpus.stopwords.words('english')
    other_words = ['rt', 'ff', 'tbt', 'ftw']  # may add more later
    stop_words.extend(other_words)
    stopped_tweet = ' '.join([w for w in tweet.split(' ') if w not in stop_words])
    return stopped_tweet


def train_model(data):
    """Train the CNN model"""

    tweet_df = pd.read_csv(data)
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: cnn_stopwords(x))
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: cnn_tokenizer(x))
    tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: cnn_preprocessor(x))
    y = tweet_df['class'].astype(int)



    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.1)

    return


if __name__ == '__main__':
    train_model("data/private/td_zw_labeled_data.csv")
    #text = "of course you were born in serbia...you're as fucked as A Serbian Film #MKR where's "
    #print(cnn_tokenizer(text))
