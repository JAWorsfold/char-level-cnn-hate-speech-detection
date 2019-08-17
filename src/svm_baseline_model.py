"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: TBC"""

import re
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
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
    re_html_entity = '&[^\s]*;' # discovered that Davidson et al. was not utf-8 encoded
    preprocessed_tweet = re.sub(re_html_entity, ' ', tweet)
    preprocessed_tweet = re.sub(re_mention, ' ', preprocessed_tweet)
    preprocessed_tweet = re.sub(re_url, ' ', preprocessed_tweet)
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


def train_model(data):
    """
    Decide whether to use over-sampling, under-sampling or SMOTE to deal with imbalance.
    Class split: 0=25863, 1=2285
    """
    data_frame = pd.read_csv(data)
    tweets = data_frame.tweet

    svm_stop_words = nltk.corpus.stopwords.words('english')
    other_words = ['rt', 'ff', 'tbt', 'ftw', 'becau']  # may add more later
    svm_stop_words.extend(other_words)
    svm_stop_words = svm_tokenizer(svm_preprocessor(' '.join(svm_stop_words)))
    vectorizer = TfidfVectorizer(
        preprocessor=svm_preprocessor,
        tokenizer=svm_tokenizer,
        stop_words=svm_stop_words,
        ngram_range=(1, 3),
        decode_error='replace',
        max_features=15000,
        max_df=0.75,
        min_df=5
    )

    X = pd.DataFrame(np.array(vectorizer.fit_transform(tweets).toarray()), dtype=np.float32)
    y = data_frame['class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.1)

    # over sampling
    X = pd.concat([X_train, y_train], axis=1)
    not_hate = X[X['class']==0]
    hate = X[X['class']==1]
    hate = resample(hate, replace=True, n_samples=len(hate)*4, random_state=33)
    X = pd.concat([not_hate, hate])
    X_train = X.drop('class', axis=1)
    y_train = X['class']

    # under sampling
    # X = pd.concat([X_train, y_train], axis=1)
    # not_hate = X[X['class']==0]
    # hate = X[X['class']==1]
    # not_hate = resample(not_hate, replace=False, n_samples=len(hate)*2, random_state=33)
    # X = pd.concat([not_hate, hate])
    # X_train = X.drop('class', axis=1)
    # y_train = X['class']

    for c in [0.01, 0.025, 0.1, 0.2, 0.25, 0.3, 1]:

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
        # plot.xlabel('Predicted Class', fontsize=12)
        # plot.show()

    # may want later for classification of my other tweets.
    # return svm

if __name__ == '__main__':
    train_model("data/private/td_zw_labeled_data.csv")
