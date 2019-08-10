"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: TBC"""

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def preprocessor(tweet):
    """

    :param tweet:
    :return:
    """
    preprocessed_tweet = None
    return preprocessed_tweet


def tokenizer(tweet):
    """

    :param tweet:
    :return:
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
    run_model()
