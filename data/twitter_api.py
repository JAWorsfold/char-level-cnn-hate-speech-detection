import tweepy
import json
#import pandas as pd

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


def get_tweet_status(tweet_id, api):
    """Invoke twitter api using 'tweepy' and return a single of status object based on id"""
    tweet_status = api.get_status(tweet_id, tweet_mode='extended')
    return tweet_status


def get_tweet_statuses(array_of_tweet_ids, api):
    """Invoke twitter api using 'tweepy' and return a list of status objects"""
    tweet_statuses = api.statuses_lookup(id_=array_of_tweet_ids, include_entities=False, trim_user=True)
    return tweet_statuses


def string_to_csv(string):
    """Convert normal string to a csv compatible string."""
    result = string
    if "\"" in result:
        result = result.replace("\"", "\"\"")
    if "," in result or "\n" in result or "\"" in result:
        result = "\"" + result + "\""
    return result


def write_tweet_status(filepath, array_of_tweet_statuses):
    """Given an array and filepath/ destination, create the file and write the array lines to the file."""
    with open(filepath, 'w') as twitter_status_data:
        for line in array_of_tweet_statuses:
            try:
                twitter_status_data.write(line)
            except UnicodeEncodeError:
                print("UnicodeEncodeError on ID: " + line.split(',')[0])


def initialize_twitter_api():
    """Initialize twitter api using global variables for OAuth access."""
    oauth = tweepy.OAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET)
    oauth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(oauth)
    return api


def get_tweets(input_file, output_file):
    """Retrieve tweet statuses one at a time and then write them to a file"""
    all_tweet_statuses = []  # empty array to store all complete lines

    twitter_api = initialize_twitter_api()
    all_tweet_ids = read_tweet_ids(input_file)

    count = 1

    for tweet_id in all_tweet_ids:
        old_line = tweet_id.split(',')
        retweet_status = None
        try:
            tweet_status = get_tweet_status(old_line[0], twitter_api)
            try:
                retweet_status = tweet_status._json["retweeted_status"]
            except Exception:
                pass
            if old_line[0] == str(tweet_status.id):
                if retweet_status is None:
                    status = str(tweet_status.full_text)
                else:
                    status = str(tweet_status._json["retweeted_status"]["full_text"])
                csv_status = string_to_csv(status)
                new_line = old_line[0] + ',' + csv_status + ',' + old_line[1]
                print(str(count) + ', ' + new_line)
                all_tweet_statuses.append(new_line)
        except Exception:  # not certain what the exact exceptions are for tweepy
            print(str(count) + ', ' + 'Tweet not available for tweet: ' + tweet_id)

        count += 1

    write_tweet_status(output_file, all_tweet_statuses)


def get_tweets_bulk():
    """Retrieve tweet statuses in bulk and then write them to a file"""
    sub_tweet_ids_only = []  # empty array to store tweet ids
    all_tweet_statuses = []  # empty array to store all complete lines

    twitter_api = initialize_twitter_api()
    all_tweet_ids = read_tweet_ids('data/public/waseem_labeled_id_data.csv')

    while all_tweet_ids:
        sub_tweet_ids = all_tweet_ids[:TWEETS_PER_REQUEST - 1]
        all_tweet_ids = all_tweet_ids[TWEETS_PER_REQUEST - 1:]
        for line in sub_tweet_ids:
            id_only = line.split(',')[0]
            sub_tweet_ids_only.append(id_only)

        sub_tweet_statuses = get_tweet_statuses(sub_tweet_ids_only, twitter_api)

        #now need to concat the status onto the original id+val line
        for i in range(0, len(sub_tweet_statuses)):
            old_line = sub_tweet_ids[i].split(',')
            if old_line[0] == sub_tweet_statuses[i].id:
                new_line = old_line[0] + ',' + sub_tweet_statuses[i].text + ',' + old_line[1]
                all_tweet_statuses.append(new_line)

        sub_tweet_ids_only.clear()

    write_tweet_status('data/private/waseem_labeled_status_data.csv', all_tweet_statuses)


if __name__ == '__main__':
    #test_pd_file = pd.read_csv('data/private/davidson_labeled_status_data.csv', nrows=15)
    #print(test_pd_file)

    get_tweets(input_file='data/public/waseem_labeled_id_data.csv',
               output_file='data/private/waseem_labeled_status_data.csv')
