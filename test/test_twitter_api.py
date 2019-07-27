from data.twitter_api import *

# test data for comparisons
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

test_real_id = ['896523232098078720,0\n',
                '896523304873238528,0\n',
                '796394920051441664,0\n']

test_real_result = ['896523232098078720,"No one is born hating another person because of the color of his skin or his background or his religion...",0\n',
                    '896523304873238528,Thank you for everything. My last ask is the same as my first. I\'m asking you to believe—not in my ability to create change, but in yours.,0\n',
                    '796394920051441664,"To all the little girls watching...never doubt that you are valuable and powerful & deserving of every chance & opportunity in the world.",0\n']

def test_read_tweet_ids():
    result_one = read_tweet_ids('test/test_data_one.txt')
    result_two = read_tweet_ids('test/test_data_two.txt')
    # will be used to test real tweets
    # result_three = read_tweet_ids('test/test_data_real.txt')
    assert result_one == test_one
    assert result_two == test_two
    #assert result_three == test_three


def test_get_tweet_status():
    api = initialize_twitter_api()
    for i in range(len(test_real_id)):
        status_object = get_tweet_status(test_real_id[i].split(',')[0], api)
        assert status_object.full_text == test_real_result[i].split(',')[1]


def test_string_to_csv():
    test_string_one = "123456789,\n,123456"
    test_string_two = "She said \"Let there be light\" \nand there was"
    test_string_three = "Once, upon, a, time,\n \"In a galaxy\" far,\nfar \"away\""

    actual_string_one = string_to_csv(test_string_one)
    actual_string_two = string_to_csv(test_string_two)
    actual_string_three = string_to_csv(test_string_three)

    expected_string_one = "\"123456789,\n,123456\""
    expected_string_two = "\"She said \"\"Let there be light\"\" \nand there was\""
    expected_string_three = "\"Once, upon, a, time,\n \"\"In a galaxy\"\" far,\nfar \"\"away\"\"\""

    assert actual_string_one == expected_string_one
    assert actual_string_two == expected_string_two
    assert actual_string_three == expected_string_three


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


def test_get_tweets():
    input_file = 'test/test_data_real.txt'
    output_file = 'test/test_write_real.txt'
    get_tweet_status(input_file, output_file)
    output_read_array = read_tweet_ids(output_file)
    assert output_read_array == test_real_result
