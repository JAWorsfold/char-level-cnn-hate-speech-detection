"""Using the command line, run this file to classify user input as a file or string"""
import argparse
from src.train_logreg import *
from src.train_svm import *
from src.train_cnn import *


def basic_preprocess(tweet):
    """Use the basic preprocessing steps to preprocess the tweets"""
    pp_tweet = tweet
    pp_tweet = pp.remove_noise(pp_tweet, mentions=True, replacement=' ')
    pp_tweet = pp.normalise(pp_tweet, numbers=True, stopwords=True, stem_words=True, replacement=' ')
    return pp_tweet


def advanced_preprocess(tweet):
    """Use the advanced preprecessing steps to preprocess the tweets"""
    pass


def load_basic_model(model_file, vectorizer_file):
    """Load the either the Logistic Regression or SVM model and vectorizer"""
    load_logreg = pickle.load(open(model_file, 'rb'))
    load_vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    print("Loading model complete")
    return load_logreg, load_vectorizer


# basic or advanced passed as parameter
def load_cnn(type):
    """Load either the normal CNN or advanced CNN and tokenizer"""
    pass


def write_results(write_path, tweets, results):
    """Write back the results to a csv file"""
    return True


def run_prediction(args):
    """Run predictions based on passed arguments"""

    to_predict = list()     # raw tweets for predictions
    pp_to_predict = list()  # preprocessed tweets for predictions
    results = list()        # results of the predictions

    # try filepath
    if args.filepath is not None:
        with open(args.filepath) as sample:
            sample_tweets = pd.read_csv(sample)
        to_predict = sample_tweets['tweet']
        to_predict = to_predict.values.tolist()
        print("File used for predictions: " + args.filepath)

    # try string
    if args.string is not None:
        to_predict.append(args.string)
        print("String to predict: " + str(to_predict))

    if not to_predict:
        raise Exception("No file path or string was provided in order to make a prediction")

    if args.model == 'cnn+' or args.model == 'cnn':
        pass

    elif args.model == 'svm' or args.model == 'logreg':
        model, vectorizer = load_basic_model("models/" + args.model + ".pickle", "models/vectorizer.pickle")
        for tweet in to_predict:
            pp_to_predict.append(basic_preprocess(tweet))
        print("Preprocessing complete.")
        to_predict_vect = vectorizer.transform(pp_to_predict)
        if args.model == 'logreg':
            results = model.predict_proba(to_predict_vect)
        else:
            results = model.predict(to_predict_vect)
        for i in range(len(results)):
            print("Prediction: " + str(results[i]) + " Tweet: " + str(pp_to_predict[i]))
    else:
        raise Exception("Only available models are 'cnn+', 'cnn', 'svm', and 'logreg'.")

    if args.write is not None:
        written = write_results(args.write, pp_to_predict, results)
        if written: print("Write back complete")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hate speech classification project. Make predictions.')

    # arguments
    parser.add_argument('-m', '--model', type=str, default='cnn+',
                        help="Choose a model to make predictions.")
    parser.add_argument('-f', '--filepath',  type=str, default=None,
                        help="Provide a file containing tweets for classification")
    parser.add_argument('-s', '--string', type=str, default=None,
                        help="Provide a single string for classification")
    parser.add_argument('-w', '--write', type=str, default=None,
                        help="Define a file path to write predictions to")
    args = parser.parse_args()

    if run_prediction(args):
        print("Program completed successfully.")
