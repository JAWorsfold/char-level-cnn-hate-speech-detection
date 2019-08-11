"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: TBC"""

import re
import string
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def svm_preprocessor(tweet):
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
    preprocessed_tweet = re.sub(re_mention, '', tweet)
    preprocessed_tweet = re.sub(re_url, '', preprocessed_tweet)
    preprocessed_tweet = re.sub(re_whitespace, ' ', preprocessed_tweet)
    return preprocessed_tweet


def svm_tokenizer(tweet):
    """
    Take a tweet - a simple text string - and remove punctuation, set characters to
    lowercase and stem the tweets.
    :param tweet: a text string
    :return tokenized_tweet: all lowercase, no punctuation and stemmed
    """
    tweet_lower = tweet.lower()
    no_punc = re.split('[^a-zA-Z ]*', tweet_lower)
    stripped = ''.join(no_punc).strip()
    stemmer = nltk.PorterStemmer()
    tokenized_tweet = [stemmer.stem(w) for w in stripped.split()]
    return tokenized_tweet


def run_model ():
    """
    Decide whether to use over-sampling, under-sampling or SMOTE to deal with imbalance.
    Class split: 0=25863, 1=2285
    :return:
    """


    #I know I'm going to use a TfIdf Vectorizer so I'll build it here and write the functions after
    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        stop_words=stopwords,
        ngram_range=(1, 3),
        decode_error='replace'
    )


if __name__ == '__main__':
    #run_model()
    test_one = ("TensorWatch: A debugging and visualization system for machine learning http://bit.ly/2KFUvqe   #AI   #DeepLearning   #MachineLearning  #DataScience")
    print(svm_tokenizer(test_one))
