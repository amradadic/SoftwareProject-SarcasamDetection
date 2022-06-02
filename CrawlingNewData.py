import tweepy
import datetime
import pandas as pd
import numpy as np


API_key = "vlY9i6EAhbLvsXkN20CaDjImR"
API_secret = "y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ"
Access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
Access_token_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"


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
max_tweets = 3000

#tweets = tw.Cursor(api.search,q=search_words,lang="en",since=date_since,until=date_until,result_type="recent").items(2)
start_date = datetime.datetime(2022, 1, 1, 00, 00, 00)
end_date = datetime.datetime(2022, 2, 1, 00, 00, 00)

for j in range (1,30):
    searched_tweets = [status for status in tweepy.Cursor(api.search_tweets,
                                                      q=querys,
                                                      tweet_mode='extended').items(max_tweets)]


    new_rows = []
    raw = []
    counter = 0
    file_counter = 0
    for i in searched_tweets:
    #print(i.user.id)
    #print("ID: ",  i.id ,   i.text)
        counter = counter + 1

        single_row = []
        single_row.append(i.id)
    #if hasattr(i, 'retweeted_status'):
        #single_row.append(i.retweeted_status.full_text)
    #else:
        single_row.append(i.full_text)

        single_row.append(i.user.id)





        new_rows.append(single_row)
        raw.append(i)


        arr1 = np.asarray(new_rows)
        string1 = "newDataset" + str(j) + ".csv"
        pd.DataFrame(arr1).to_csv(string1)
        string2 = "raw" + str(j) + ".csv"
        arr2 = np.asarray(raw)
        pd.DataFrame(arr2).to_csv(string2)


