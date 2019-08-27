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
    print(test_data)
    assert True


def test_normalise(pre_processor):
    assert True


def test_tokenize(pre_processor):
    assert True

