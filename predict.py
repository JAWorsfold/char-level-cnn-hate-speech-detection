"""Using the command line, run this file to classify user input as a file or string"""
import argparse
import pickle


def load_logreg(model_file, vectorizer_file):
    """Load the Logistic Regression model and vectorizer"""
    load_logreg = pickle.load(open(model_file, 'rb'))
    load_vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    return load_logreg, load_vectorizer

def load_svm():
    pass


# basic or advanced passed as parameter
def load_cnn(type):
    pass


def run_prediction(args):
    """Run predictions based on passed arguments"""

    to_predict = list()

    # try filepath
    if args.filepath is not None:
        pass

    # try string
    if args.string is not None:
        pass

    if not to_predict:
        raise Exception("No filepath or string was provided in order to make a prediction")

    if args.model == 'cnn+':
        pass
    elif args.model == 'cnn':
        pass
    elif args.model == 'svm':
        pass
    elif args.model == 'logreg':
        model, vectorizer = load_logreg("models/logreg.pickle", "models/vectorizer.pickle")
    else:
        raise Exception("Only available models are 'cnn+', 'cnn', 'svm', and 'logreg'.")

    # make prediction(s)


    # return prediction as a list or double. OR just print them to the screen.
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hate speech classification project. Make predictions.')

    # arguments
    parser.add_argument('-m', '--model', type=str, defaul='cnn+',
                        help="Choose a model to make predictions.")
    parser.add_argument('-f', '--filepath',  type=str, default=None,
                        help="Provide a file containing tweets for classification")
    parser.add_argument('-s', '--string', type=str, default=None,
                        help="Provide a single string for classification")



    args = parser.parse_args()
    run_prediction(args)
