"""Using the command line run this file to classify user input"""
import argparse


def load_logreg():
    pass


def load_svm():
    pass


# basic or advanced passes as parameter
def load_cnn(type):
    pass


def run_prediction(args):
    #try filepath

    #try string

    # try load model

    # make prediction(s)

    # return prediction as a list or double. OR just print them to the screen.
    pass


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
