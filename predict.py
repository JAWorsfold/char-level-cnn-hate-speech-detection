"""Using the command line, run this file to classify user input as a file or string"""
import argparse
import csv
from src.train_logreg import *
from src.train_svm import *
# from src.train_cnn import *


pp = PreProcessUtils()


def basic_preprocess(tweet):
    """Use the basic preprocessing steps to preprocess the tweets"""
    more_stopwords = ['rt', 'ff', 'tbt', 'ftw', 'dm']  # may add more later
    pp_tweet = tweet
    pp_tweet = pp.remove_noise(pp_tweet, mentions=True, replacement=' ')
    pp_tweet = pp.normalise(pp_tweet, numbers=True, stopwords=True, other_stopwords=more_stopwords,
                            stem_words=True, replacement=' ')
    return pp_tweet


def advanced_preprocess(tweet):
    """Use the advanced preprecessing steps to preprocess the tweets"""
    pp_tweet = tweet
    pp_tweet = pp.remove_noise(pp_tweet, mentions=False, replacement='')
    pp_tweet = pp.normalise(pp_tweet, punctuation=False, numbers=False, stopwords=False, stem_words=False, replacement='')
    return pp_tweet


def load_basic_model(model_file, vectorizer_file):
    """Load the either the Logistic Regression or SVM model and vectorizer"""
    load_model = pickle.load(open(model_file, 'rb'))
    load_vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    print("Loading model complete")
    return load_model, load_vectorizer


# basic or advanced passed as parameter
def load_cnn(type):
    """Load either the normal CNN or advanced CNN and tokenizer"""
    pass


def write_results(write_path, final_predictions):
    """Write back the results to a csv file"""
    with open(write_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(final_predictions)
    return True


def run_prediction(args):
    """Run predictions based on passed arguments"""

    to_predict = list()        # raw tweets for predictions
    pp_to_predict = list()     # preprocessed tweets for predictions
    final_predictions = list() # predictions and tweets for publishing

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
        model, vectorizer = load_basic_model("models/" + args.model + ".pickle",
                                             "models/" + args.model + "_vectorizer.pickle")
        for tweet in to_predict:
            pp_to_predict.append(basic_preprocess(tweet))
        print("Preprocessing complete.")
        to_predict_vect = vectorizer.transform(pp_to_predict)
        results = model.predict(to_predict_vect)
        if args.model == 'logreg':
            results_prob = model.predict_proba(to_predict_vect)
            for i in range(len(results)):
                final_predictions.append([str(results[i]), str(results_prob[i][1]), str(to_predict[i])])
        else:
            for i in range(len(results)):
                final_predictions.append([str(results[i]), str(to_predict[i])])
    else:
        raise Exception("The only available models are 'cnn+', 'cnn', 'svm', and 'logreg'.")

    for p in final_predictions:
        print(p)

    if args.write is not None:
        write_file_path = args.write + "/" + args.model + "_predictions.csv"
        written = write_results(write_file_path, final_predictions)
        if written: print("Write back complete")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hate speech classification project. Make predictions.')

    # arguments
    parser.add_argument('-m', '--model', type=str, default='cnn+',
                        help="Choose a model to make predictions: 'cnn+', 'cnn', 'logreg', 'svm'")
    parser.add_argument('-f', '--filepath',  type=str, default=None,
                        help="Provide a file containing tweets for classification")
    parser.add_argument('-s', '--string', type=str, default=None,
                        help="Provide a single string for classification")
    parser.add_argument('-w', '--write', type=str, default=None,
                        help="Define a file path to write predictions to")
    args = parser.parse_args()

    if run_prediction(args):
        print("Program completed successfully.")
