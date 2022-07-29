import csv
import tweepy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Oauth keys
consumer_key = "vlY9i6EAhbLvsXkN20CaDjImR"
consumer_secret = "y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ"
access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
access_token_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"

API_key = "vlY9i6EAhbLvsXkN20CaDjImR"
API_secret = "y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ"
Access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
Access_token_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"


# Authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# update these for the tweet you want to process replies to 'name' = the account username and you can find the tweet id within the tweet URL
name = 'Yankees'
tweet_id = '758087956754161664'

replies=[]
for tweet in tweepy.Cursor(api.search_tweets,q='to:'+name, result_type='recent', timeout=999999).items(1000):
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        if (tweet.in_reply_to_status_id_str==tweet_id):
            replies.append(tweet)

with open('replies_clean.csv', 'w') as f:
    csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
    csv_writer.writeheader()
    for tweet in replies:
        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
        csv_writer.writerow(row)