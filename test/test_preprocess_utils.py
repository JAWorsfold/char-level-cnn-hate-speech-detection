from src.preprocess_utils import *
import csv
import pytest


@pytest.fixture()
def pre_processor():
    pp = PreProcessUtils()  # use default instance variables
    return pp


@pytest.fixture()
def test_data():
    with open('test/preprocess_data/test_input.csv') as test_data:
        csv_data = csv.reader(test_data)
        list_data = list(csv_data)
    return list_data


def test_remove_noise(pre_processor, test_data):
    with open('test/preprocess_data/noise_removed.csv') as expected:
        exp_lst = list(csv.reader(expected))
    for i in range(len(exp_lst)):
        expected = exp_lst[i]
        actual = pre_processor.remove_noise(test_data[i], mentions=True, replacement=' ')
        assert actual == expected


def test_normalise_stopwords_false(pre_processor, test_data):
    with open('test/preprocess_data/normalised.csv') as expected:
        exp_lst = list(csv.reader(expected))
    for i in range(len(exp_lst)):
        expected = exp_lst[i]
        actual = pre_processor.normalise(test_data[i], numbers=True, stopwords=False, replacement=' ')
        assert actual == expected


def test_normalise_stopwords_true(pre_processor, test_data):
    with open('test/preprocess_data/normalised.csv') as expected:
        exp_lst = list(csv.reader(expected))
    add_stopwords = ['rt']
    for i in range(len(exp_lst)):
        expected = exp_lst[i]
        actual = pre_processor.normalise(test_data[i], numbers=True, stopwords=True,
                                         other_stopwords=add_stopwords, replacement=' ')
        assert actual == expected


def test_tokenize(pre_processor, test_data):
    with open('test/preprocess_data/tokenized.csv') as expected:
        exp_lst = list(csv.reader(expected))
    for i in range(len(exp_lst)):
        expected = exp_lst[i]
        actual = pre_processor.tokenize(test_data[i])
        assert ' '.join(actual) == expected
