"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: TBC"""

import re
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
import seaborn as sns


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
    """

    data_frame = pd.read_csv("data/private/td_zw_labeled_data.csv")
    tweets = data_frame.tweet

    svm_stop_words = nltk.corpus.stopwords.words('english')
    other_words = ['rt']  # may add more later
    svm_stop_words.extend(other_words)
    svm_stop_words = svm_tokenizer(svm_preprocessor(' '.join(svm_stop_words)))
    vectorizer = TfidfVectorizer(
        preprocessor=svm_preprocessor,
        tokenizer=svm_tokenizer,
        stop_words=svm_stop_words,
        ngram_range=(1, 3),
        decode_error='replace'
    )

    vectorizer.fit(tweets)
    X = vectorizer.transform(tweets)
    y = data_frame['class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2)

    for c in [0.25, 0.5, 1, 2.5, 5]:

        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        results = classification_report(y_test, y_pred)
        print('####################### C=' + str(c) + ' #######################')
        print(results)
        print()

        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        # names = ['Non-hate', 'Hate']
        # df_cm = pd.DataFrame(conf_matrix, index=names, columns=names)
        # plot.figure(figsize=(6,6))
        # sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', square=True, fmt='.2f')
        # plot.ylabel('Actual Class', fontsize=12)
        # plot.xlabel('Predicted Classs', fontsize=12)
        # plot.show()


if __name__ == '__main__':
    run_model()
