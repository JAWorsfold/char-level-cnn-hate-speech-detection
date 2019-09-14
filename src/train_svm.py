"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: Automated Hate Speech Detection and the Problem of Offensive Language.
Source: https://github.com/t-davidson/hate-speech-and-offensive-language"""

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.preprocess_utils import PreProcessUtils


PP = PreProcessUtils()


def svm_preprocessor(tweet):
    """
    Convert tweet to lowercase, stem words and remove or replace:
    1) Mentions i.e. @someone, 2) URLs i.e. https://www.twitter.com, 3) HTML entitiesRepeated,
    4) whitespace, 5) punctuation, 6) number, 7) stop words
    See PreProcessUtils Class for more details.
    :param tweet: text string
    :return pp_tweet: A pre-processed tweet
    """
    pp_tweet = tweet
    pp_tweet = PP.remove_noise(pp_tweet, mentions=True, replacement=' ')
    pp_tweet = PP.normalise(pp_tweet, numbers=True, stopwords=True, stem_words=True, replacement=' ')
    return pp_tweet


def svm_tokenizer(tweet):
    """Tokenize the tweet"""
    return PreProcessUtils.tokenize(tweet)


def train_model(data):
    """
    Decide whether to use over-sampling, under-sampling or SMOTE to deal with imbalance.
    Class split: 0=25863, 1=2285
    """
    tweet_df = pd.read_csv(data)

    more_stopwords = ['rt', 'ff', 'tbt', 'ftw', 'dm']  # may add more later
    svm_stop_words = PP.stop_words(more_stopwords)

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

    X = pd.DataFrame(np.array(vectorizer.fit_transform(tweet_df.tweet).toarray()), dtype=np.float32)
    y = tweet_df['class'].astype(int)
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

    # for c in [0.01, 0.025, 0.1, 0.2, 0.25, 0.3, 1]:

    svm = LinearSVC(C=0.1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results = classification_report(y_test, y_pred)
    print('####################### C=' + str(0.1) + ' #######################')
    print(results)
    print()

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # may want later for classification of my other tweets.
    return svm, vectorizer


if __name__ == '__main__':
    # train_model('data/private/td_zw_labeled_data.csv')
    model, vectorizer = train_model('data/private/td_zw_labeled_data.csv')

    pickle.dump(model, open("models/svm.pickle", "wb"))
    pickle.dump(vectorizer, open("models/svm_vectorizer.pickle", "wb"))

    test_pred_tweet = ["Listen up women, there is a reason why god made men stronger than you. It's "
                       "because rape is a force of good, that should be celebrated. IF a man wants to "
                       "fill you with his big cock, you should learn to : relax, orgasm and thank him!!! "
                       "Heil Hitler #bussdown",
                       "To all the little girls watching...never doubt that you are valuable and powerful "
                       "&amp; deserving of every chance &amp; opportunity in the world."]

    test_pred = vectorizer.transform(test_pred_tweet)
    pred_result = model.predict(test_pred)
    print("test = %s, Predicted = %s" % (test_pred, pred_result))
