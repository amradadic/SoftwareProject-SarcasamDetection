import re
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
    response = requests.get(url, auth=bearer_oauth, params=params)

    while response.status_code == 429:
        timeout = 100  #15min
        print(f'Sleeping for {timeout} seconds')
        time.sleep(timeout)
        response = requests.get(url, auth=bearer_oauth, params=params)
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


def recognizeTypeOfTweet(thread_pattern, list):

    index_list = []
    if len(list) == 0:
        return index_list

    tweet_text =  list[0][1]
    lice = 0
    if re.match("(A)([^A]*)(A)([^A]*)$", thread_pattern) and (('I was' in tweet_text) or ('I am' in tweet_text) or ('I\'m' in tweet_text)):

        index_of_A = ([pos for pos, char in enumerate(thread_pattern) if char == 'A'])

        #cue, oblivious, sarcastic, elicit
        index_list = [0, 1, index_of_A[1], index_of_A[1] + 1]

    elif re.match("(A)A*(B)(A*)$", thread_pattern) and  (('you\'re') in tweet_text
                                                         or ('you are') in tweet_text
                                                         or ('are you') in tweet_text
                                                         or ('you were') in tweet_text
                                                         or ('were you') in tweet_text):

        index_of_B = ([pos for pos, char in enumerate(thread_pattern) if char == 'B'])

        # cue, oblivious, sarcastic, elicit
        index_list = [0, -1, index_of_B[0], index_of_B[0] + 1]

    elif re.match("(A)(A*B[AB]*)(C)([AB]*)$", thread_pattern) and (('she is') in tweet_text
                                                                   or ('she was') in tweet_text
                                                                   or ('was she') in tweet_text
                                                                   or ('she\'s') in tweet_text
                                                                   or ('he is') in tweet_text
                                                                   or ('he was') in tweet_text
                                                                   or ('was he') in tweet_text
                                                                   or ('he\'s') in tweet_text
                                                                   or ('they were') in tweet_text
                                                                   or ('they\'re') in tweet_text
                                                                   or ('were they') in tweet_text
                                                                   or ('they are') in tweet_text):

        index_of_C = ([pos for pos, char in enumerate(thread_pattern) if char == 'C'])
        # cue, oblivious, sarcastic, elicit
        index_list = [0, 1, index_of_C[0], index_of_C[0] + 1]

    return index_list

def defineExpressionsForTweet():
    df = pd.read_csv('filtered.csv')
    for i in range(204, 210):  #df.shape[0]
        replies_list = getThreadRecursively(df.at[i,'tweet_id'])
        s = ''
        #replies_list.reverse()
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

        index_list_types = recognizeTypeOfTweet(s, replies_list)

        print(index_list_types, s)



        #print(s, " ", df.at[i,'tweet_id'])
        df.loc[i, 'pattern'] = s

        if i % 10 == 0:
            df.to_csv("filtered.csv", index=False)
            print("Wrote to csv ", i)

    return

defineExpressionsForTweet()








