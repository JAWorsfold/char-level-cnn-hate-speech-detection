"""This model is based partly on a similar model used by Thomas Davidson et al. in
his paper: Automated Hate Speech Detection and the Problem of Offensive Language.
Source: https://github.com/t-davidson/hate-speech-and-offensive-language"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.preprocess_utils import PreProcessUtils
import pickle


PP = PreProcessUtils()


def lr_preprocessor(tweet):
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


def lr_tokenizer(tweet):
    """Tokenize the tweet"""
    return PreProcessUtils.tokenize(tweet)


def train_model(data):
    """Train the model"""
    tweet_df = pd.read_csv(data)

    more_stopwords = ['rt', 'ff', 'tbt', 'ftw']  # may add more later
    lr_stop_words = PP.stop_words(more_stopwords)

    vectorizer = TfidfVectorizer(
        preprocessor=lr_preprocessor,
        tokenizer=lr_tokenizer,
        stop_words=lr_stop_words,
        ngram_range=(1, 3),
        decode_error='replace',
        max_features=15000,
        max_df=0.75,
        min_df=5
    )

    X = pd.DataFrame(np.array(vectorizer.fit_transform(tweet_df.tweet).toarray()), dtype=np.float32)
    y = tweet_df['class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.1)

    c = [0.01, 0.1]

    parameters = [{"penalty": ["l1"], "C": c, "solver": ["saga", "liblinear"]},
                  {"penalty": ["l2"], "C": c, "solver": ["saga", "sag"]}]

    classifier = GridSearchCV(LogisticRegression(class_weight="balanced",random_state=33),
                              parameters,
                              scoring="f1_micro",
                              cv=StratifiedKFold(n_splits=5, random_state=33).split(X_train, y_train),
                              verbose=2)

    classifier.fit(X_train, y_train)

    print("best paramaters: ")
    print(classifier.best_params_)
    print("best estimator: ")
    print(classifier.best_estimator_)
    print("best score: ")
    print(classifier.best_score_)

    print("train best")
    lr = classifier.best_estimator_
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results = classification_report(y_test, y_pred)
    print(results)
    print()

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    return lr, vectorizer

if __name__ == '__main__':
    # train_model('data/private/td_zw_labeled_data.csv')
    model, vectorizer = train_model('data/private/td_zw_labeled_data.csv')

    pickle.dump(model, open("models/logreg.pickle", "wb"))
    pickle.dump(vectorizer, open("models/vectorizer.pickle", "wb"))

    test_pred_tweet = ["i hate handicap faggots"]
    test_pred = vectorizer.transform(test_pred_tweet)
    pred_result = model.predict(test_pred)

    print("test = %s" % test_pred)
    print("model.predict = %s" % pred_result)

    pred_result = model.predict_log_proba(test_pred)
    print("model.predict_log_proba = %s" % pred_result)

    pred_result = model.predict_proba(test_pred)
    print("model.predict_proba = %s" % pred_result)
