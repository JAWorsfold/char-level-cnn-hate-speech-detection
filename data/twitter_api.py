import tweepy
import json

# Generated by creating a Twitter APP here: https://developer.twitter.com/en/apps
CONSUMER_API_KEY = 'to be assigned - config.json'
CONSUMER_API_SECRET = 'to be assigned - config.json'
ACCESS_TOKEN = 'to be assigned - config.json'
ACCESS_TOKEN_SECRET = 'to be assigned - config.json'

# Twitter currently allows only 100 Tweets per request:
# https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-lookup
TWEETS_PER_REQUEST = 100


#def read_


def main():
    f = open("test.txt", "r")
    for s in f:
        print(s)
    f.close()

if __name__ == '__main__':
    main()
    print("Done")
    print("Right")
