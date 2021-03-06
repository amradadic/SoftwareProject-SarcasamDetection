import tweepy
import csv

file = open('iSarcasmIDs.csv')
csvreader = csv.reader(file)

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

#Iterating through the csv file
new_rows = []
for row in csvreader:
    id = row[0]
    single_row = []
    single_row.append(id)
    single_row.append(row[1])
    counter = counter + 1

    try:
        #reading the tweet
        status = api.get_status(id, tweet_mode="extended")
        text = status.full_text
        single_row.append(text)
        single_row.append(row[2])
    except:
        single_row.append("/")
        single_row.append(row[2])

    new_rows.append(single_row)

import pandas as pd
import numpy as np
arr = np.asarray(new_rows)
pd.DataFrame(arr).to_csv("iSarcasmDataset.csv")





