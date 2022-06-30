import requests
import pandas as pd
import time
import os
import json

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAI3qcQEAAAAA4PAEfL%2ByQCQVQJpHQdNdCoBtlS0%3DvNGdgosAW0lrgv5t8alZ2buQSRfdIYVNT4KCw1Xyvql09qulAS'

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params =params)
    #print(response.status_code)
    if response.status_code == 429:
        timeout = 900 #15min
        print(f'Sleeping for {timeout} seconds')
        time.sleep(timeout)
        connect_to_endpoint(url, params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def getSingleRecursively(tweet_id):
    search_url = "https://api.twitter.com/2/tweets"

    expansions = 'author_id,in_reply_to_user_id,referenced_tweets.id'
    fields = 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'
    userfields = 'name,username'
    query_params = {'ids': tweet_id, 'tweet.fields': fields, 'expansions': expansions, 'user.fields': userfields}
    json_response = connect_to_endpoint(search_url, query_params)

    list = []
    list.append(tweet_id)
    return json_response  #(json_response['data'][0]['referenced_tweets'][0]['id'])


def getThreadRecursively(tweet_id):
    list = []
    json = getSingleRecursively(tweet_id)
    while True:
        if not 'data' in json:
            return []

        elif json['data'][0]['conversation_id'] == json['data'][0]['id']:
            list.append((json['data'][0]['id'],
                         json['data'][0]['text'],
                         json['data'][0]['author_id'],
                         json['data'][0]['created_at']))
            return list
        else:
            list.append((json['data'][0]['id'],
                         json['data'][0]['text'],
                         json['data'][0]['author_id'],
                         json['data'][0]['created_at']))
            json = getSingleRecursively(json['data'][0]['referenced_tweets'][0]['id'])

def next_alpha(s):
    return chr((ord(s.upper())+1 - 65) % 26 + 65)

def defineExpressionsForTweet():
    df = pd.read_csv('filtered.csv')
    for i in range(110, df.shape[0]):
        replies_list = getThreadRecursively(df.at[i,'tweet_id'])
        s = ''
        replies_list.reverse()
        author_id_list = {}
        letter = 'A'
        for x in replies_list:
            current_id = x[2]
            if current_id not in author_id_list:
                author_id_list[current_id] = letter
                s += letter
                letter = next_alpha(letter)
            else:
                s += author_id_list[current_id]

        #print(s, " ", df.at[i,'tweet_id'])
        df.loc[i, 'pattern'] = s

        if i % 10 == 0:
            df.to_csv("filtered.csv", index=False)
            print("Wrote to csv ", i)

    return

defineExpressionsForTweet()








