import tweepy, csv
import pandas as pd
import time


# API keyws that yous saved earlier - consumer
api_key = 'vlY9i6EAhbLvsXkN20CaDjImR'
api_secrets = 'y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ'
#Authentication
access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
access_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key, api_secrets)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')




def scrape_user_tweets(user_id, max_tweets):
    # Creation of query method using parameters
    tweets = tweepy.Cursor(api.user_timeline, user_id=user_id).items(max_tweets)

    # List comprehension pulling chosen tweet information from tweets iterable object
    # Add or remove tweet information you want in the below list comprehension
    tweets_list = [[tweet.text, tweet.created_at, tweet.id_str, tweet.user.screen_name, tweet.coordinates,
                    tweet.place, tweet.retweet_count, tweet.favorite_count, tweet.lang,
                    tweet.source, tweet.in_reply_to_status_id_str,
                    tweet.in_reply_to_user_id_str, tweet.is_quote_status,
                    ] for tweet in tweets]

    # Creation of dataframe from tweets_list
    # Add or remove columns as you remove tweet information
    tweets_df = pd.DataFrame(tweets_list,
                             columns=['Tweet Text', 'Tweet Datetime', 'Tweet Id', 'Twitter @ Name', 'Tweet Coordinates',
                                      'Place Info',
                                      'Retweets', 'Favorites', 'Language', 'Source', 'Replied Tweet Id',
                                      'Replied Tweet User Id Str', 'Quote Status Bool'])

    tweets_df.to_csv('{}-tweets.csv'.format(user_id), sep=',', index=False)


#tweet id
#id =1086119014131208193 #primjer 1
id = 710122230911520768 #primjer 2
status = api.get_status(id)
print(status.user.name)
print(status.user.screen_name)
print(status.user.location)
#id korisnika na twittweru
user_id = status.user.id
max_tweets = 2

#only for one user
scrape_user_tweets(user_id,max_tweets)




def get_all_tweet_ids():
    df = pd.read_csv('iSarcasmIDs.csv')
    tweetid_fromCsv = df['tweet_id'] #ovdje treba samo kolona u kojoj je tweet id

    return tweetid_fromCsv


def get_user_ids(tweet_ids):
    #tweet_ids is a list of tweet_id-s that we have in a csv file
    user_ids = [] #list of user id-s that we will use later
    for tweet_id in tweet_ids:
        try:
            # reading the tweet
            print("\n tweet id is")
            print(tweet_id)
            status = api.get_status(tweet_id)
            print("\nuser id is")
            print(status.user.id)
            user_ids.extend(status.user.id)
        except:
            print("\n144-not found 404")
            continue

    return user_ids


#tweetid = get_all_tweet_ids()
#user_id_list = get_user_ids(tweetid)

#for user in user_id_list:
#    print(user)




def scrape_multiple_users(username_ids, max_tweets_per):
    # Creating master list to contain all tweets
    master_tweets_list = []

    # Looping through each username in user list
    for username_id in username_ids:
        # Creation of query method using parameters
        tweets = tweepy.Cursor(api.user_timeline, user_id=username_id).items(max_tweets_per)

        # List comprehension pulling chosen tweet information from tweets iterable object
        # Appending new tweets per user into the master tweet list
        # Add or remove tweet information you want in the below list comprehension
        for tweet in tweets:
            master_tweets_list.append(
                (tweet.text, tweet.created_at, tweet.id_str, tweet.user.screen_name, tweet.coordinates,
                 tweet.place, tweet.retweet_count, tweet.favorite_count, tweet.lang,
                 tweet.source, tweet.in_reply_to_status_id_str,
                 tweet.in_reply_to_user_id_str, tweet.is_quote_status))

    # Creation of dataframe from tweets_list
    # Add or remove columns as you remove tweet information
    tweets_df = pd.DataFrame(master_tweets_list,
                             columns=['Tweet Text', 'Tweet Datetime', 'Tweet Id', 'Twitter @ Name', 'Tweet Coordinates',
                                      'Place Info',
                                      'Retweets', 'Favorites', 'Language', 'Source', 'Replied Tweet Id',
                                      'Replied Tweet User Id Str', 'Quote Status Bool'])

    # Checks if there are coordinates attached to tweets, if so extracts them

    # Checks if there is place information available, if so extracts them

    # Uncomment/comment below lines to decide between creating csv or excel file
    tweets_df.to_csv('multi-user-tweets.csv', sep=',', index=False)

#test 3 random user ids
user_id_list = [870032416685117441, 88317267, 4705188576]
max_tweets_per = 2

scrape_multiple_users(user_id_list, max_tweets_per)