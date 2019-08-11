"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: TBC"""

import re
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

    preprocessed_tweet = None
    return preprocessed_tweet


def svm_tokenizer(tweet):
    """
    Take a tweet - a simple text string - and remove punctuation, set characters to
    lowercase and stem the tweets.
    :param tweet: a text string
    :return tokenized_tweet: all lowercase, no punctuation and stemmed
    """
    tokenized_tweet = None
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
    test_one = "\"No one is born hating another person because of the color of his skin or his background or his religion...\"   "
    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(t) for t in test_one.split()]
    string_tokens = ' '.join(tokens)
    print(tokens)
    print(string_tokens)
