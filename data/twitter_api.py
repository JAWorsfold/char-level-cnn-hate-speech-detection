import tweepy
import json

# Generated by creating a Twitter APP here: https://developer.twitter.com/en/apps
with open('data/config.json') as json_file:
    config = json.load(json_file)

CONSUMER_API_KEY    = config["twitter_api"]["cons_api_key"]
CONSUMER_API_SECRET = config["twitter_api"]["cons_api_sec"]
ACCESS_TOKEN        = config["twitter_api"]["access_token"]
ACCESS_TOKEN_SECRET = config["twitter_api"]["access_secret"]

# Twitter currently allows only 100 Tweets per request:
# https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-lookup
TWEETS_PER_REQUEST = 100


def read_tweet_ids(filepath):
    """Read the contents of a file provided as a parameter and return an array of line contents."""
    with open(filepath) as twitter_id_data:
        lines = twitter_id_data.readlines()
    return lines


def get_tweet_statuses(array_of_tweet_ids, batchsize):
    """to do"""
    return []


def write_tweet_status(filepath, array_of_tweet_statuses):
    """Given an array and filepath/destination, create the file and write the array lines to the file."""
    with open(filepath, 'w') as twitter_status_data:
        for line in array_of_tweet_statuses:
            twitter_status_data.write(line)


def main():
    """Handle the bulk of the execution"""
    print(read_tweet_ids('test/test_data_one.txt'))
    return


if __name__ == '__main__':
    main()
