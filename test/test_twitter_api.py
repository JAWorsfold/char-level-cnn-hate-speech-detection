from data.twitter_api import *

#fake test data for comparisons
test_one = ['123456789,1\n',
            '987654321,0\n',
            '485020474,1\n']

test_two = ['234123513245,0\n',
            '123413462456,1\n',
            '134523424564,1\n',
            '655782456245,0\n',
            '134545734123,1\n',
            '987456735664,0\n',
            '456451237793,0\n',
            '548701896356,0\n',
            '564918597520,1\n',
            '657456793278,1\n']

test_three = ['1,0\n',
              '2,1\n',
              '3,0\n',
              '4,0\n',
              '5,1\n',
              '6,0\n',
              '7,1\n',
              '8,0\n',
              '9,0\n',
              '10,1\n',
              '11,0\n',
              '12,1\n',
              '13,0\n',
              '14,0\n',
              '15,0\n']


def test_read_tweet_ids():
    result_one = read_tweet_ids('test\test_data_one.txt')
    result_two = read_tweet_ids('test\test_data_two.txt')
    result_three = read_tweet_ids('test\test_data_three.txt')
    assert result_one == test_one
    assert result_two == test_two
    assert result_three == test_three


def test_get_tweet_statuses():
    assert True


def test_write_tweet_status():
    assert True
