import tweepy
import datetime
import pandas as pd
import numpy as np


#put api keys here
API_key = "vlY9i6EAhbLvsXkN20CaDjImR"
API_secret = "y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ"
Access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
Access_token_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"

#put csv name files here
output_file = "data.csv"  #main file used later for filtering
raw_data = "raw.csv"  #raw file where everything is saved for keeping

#amount of recent tweets crawled at a single time
max_tweets = 100


auth = tweepy.OAuthHandler(API_key, API_secret)
auth.set_access_token(Access_token, Access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True,
                        retry_count=10, retry_delay=60,
                        retry_errors=[400] + list(range(402,599)))

try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')

querys = 'being sarcastic -filter:retweets AND -filter:links AND -filter:media'

searched_tweets = [status for status in tweepy.Cursor(api.search_tweets,
                                                      q=querys,
                                                      tweet_mode='extended').items(max_tweets)]


new_rows = []
raw = []
counter = 0
file_counter = 0
for i in searched_tweets:

    counter = counter + 1

    single_row = []
    single_row.append(i.id)
    single_row.append(i.full_text)
    single_row.append(i.user.id)

    new_rows.append(single_row)
    raw.append(i)


formated_array = np.asarray(new_rows)
df = pd.DataFrame(formated_array)
df = df.rename(columns={0:"tweet_id", 1:"text", 2:"user_id"})
df.to_csv(output_file)
raw_array = np.asarray(raw)
pd.DataFrame(raw_array).to_csv(raw_data)


