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
                '796394920051441664,0\n']

test_real_result = ['896523232098078720,\"\"\"No one is born hating another person because of the color of his skin or his background or his religion...\"\" https://t.co/InZ58zkoAm\",0\n',
                    '796394920051441664,\"\"\"To all the little girls watching...never doubt that you are valuable and powerful &amp; deserving of every chance &amp; opportunity in the world.\"\"\",0\n']

def test_read_tweet_ids():
    result_one = read_tweet_ids('test/api_data/input_one.csv')
    result_two = read_tweet_ids('test/api_data/input_two.csv')
    result_three = read_tweet_ids('test/api_data/input_real.csv')
    assert result_one == test_one
    assert result_two == test_two
    assert result_three == test_real_id


def test_get_tweet_status():
    api = initialize_twitter_api()
    for i in range(len(test_real_id)):
        status_object = get_tweet_status(test_real_id[i].split(',')[0], api)
        actual_string = string_to_csv(status_object.full_text)
        expected_string = test_real_result[i].split(',')[1]
        assert actual_string == expected_string


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
    write_tweet_status('test/api_data/output_one.csv',test_one)
    write_tweet_status('test/api_data/output_two.csv',test_two)
    #write_tweet_status(test_three)

    with open('test/api_data/output_one.csv') as test_file:
        test_file_one = test_file.readlines()
    with open('test/api_data/output_two.csv') as test_file:
        test_file_two = test_file.readlines()

    assert test_file_one == test_one
    assert test_file_two == test_two


def test_get_tweets():
    input_file = 'test/api_data/input_real.csv'
    output_file = 'test/api_data/output_real.csv'
    get_tweets(input_file, output_file)
    output_read_array = read_tweet_ids(output_file)
    assert output_read_array == test_real_result
