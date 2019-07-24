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

test_real = ['896523232098078720,0,"No one is born hating another person because of the color of his skin or his background or his religion..."\n',
             '896523304873238528,0,Thank you for everything. My last ask is the same as my first. I\'m asking you to believe—not in my ability to create change, but in yours.\n',
             '796394920051441664,0,"To all the little girls watching...never doubt that you are valuable and powerful & deserving of every chance & opportunity in the world."\n']

def test_read_tweet_ids():
    result_one = read_tweet_ids('test/test_data_one.txt')
    result_two = read_tweet_ids('test/test_data_two.txt')
    # will be used to test real tweets
    # result_three = read_tweet_ids('test/test_data_real.txt')
    assert result_one == test_one
    assert result_two == test_two
    #assert result_three == test_three


def test_get_tweet_statuses():
    assert True


def test_write_tweet_status():
    write_tweet_status('test/test_write_one.txt',test_one)
    write_tweet_status('test/test_write_two.txt',test_two)
    #write_tweet_status(test_three)

    with open('test/test_write_one.txt') as test_file:
        test_file_one = test_file.readlines()
    with open('test/test_write_two.txt') as test_file:
        test_file_two = test_file.readlines()

    assert test_file_one == test_one
    assert test_file_two == test_two
