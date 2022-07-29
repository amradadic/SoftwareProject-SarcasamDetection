import re
import requests
import pandas as pd
import time


#put the filtered cue tweets dataset
filtered = "filtered.csv"

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAModewEAAAAAXDLH1uGk8WWWG4uybT%2BXx5heUFA%3DFVHX36WTXcC9SQlFtCSHtKvnEs2jnsqwsHOloDx05M50WuoVM7'


#authetication
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

#connection and exception handling
def connect_to_endpoint(url, params):

    response = requests.get(url, auth=bearer_oauth, params=params)

    while response.status_code == 429:
        timeout = 1000  #sleep for 15min
        print(f'Sleeping for {timeout} seconds')
        time.sleep(timeout)
        response = requests.get(url, auth=bearer_oauth, params=params)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


#getting a single tweet and its full json result
def get_single_tweet(tweet_id):

    search_url = "https://api.twitter.com/2/tweets"
    expansions = 'author_id,in_reply_to_user_id,referenced_tweets.id'
    fields = 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'
    userfields = 'name,username'
    query_params = {'ids': tweet_id, 'tweet.fields': fields, 'expansions': expansions, 'user.fields': userfields}
    json_response = connect_to_endpoint(search_url, query_params)
    return json_response

#getting the thread of a tweet and returning a formatted list
def get_thread_recursively(tweet_id):
    list = []
    json = get_single_tweet(tweet_id)

    while True:
        #if a tweet in thread is deleted - return empty
        if not 'data' in json:
            return []

        #if the tweet is the first tweet in conversation, finish
        elif json['data'][0]['conversation_id'] == json['data'][0]['id']:
            list.append((json['data'][0]['id'],
                         json['data'][0]['text'],
                         json['includes']['users'][0]['username'] + '|' + json['data'][0]['author_id'],
                         json['data'][0]['created_at']))

            return list

        #getting the next tweet recursively
        else:
            list.append((json['data'][0]['id'],
                         json['data'][0]['text'],
                         json['includes']['users'][0]['username'] + '|' + json['data'][0]['author_id'],
                         json['data'][0]['created_at']))
            json = get_single_tweet(json['data'][0]['referenced_tweets'][0]['id'])


#next letter of the alphabet
def next_alpha(s):
    return chr((ord(s.upper())+1 - 65) % 26 + 65)

#pattern matching and marking the tweets
def recognize_type_of_tweet(thread_pattern, list):

    index_list = []

    #if list is empty return
    if len(list) == 0:
        return index_list

    tweet_text = list[0][1]

    if re.match("(A)([^A]*)(A)([^A]*)$", thread_pattern) and (('i was' in tweet_text.lower())
                                                              or ('i am' in tweet_text.lower())
                                                              or ('i\'m' in tweet_text.lower())):

        index_of_A = ([pos for pos, char in enumerate(thread_pattern) if char == 'A'])

        #putting the position of type of tweet in the index list that is returned
        #cue, oblivious, sarcastic, elicit, person class (1, 2 or 3)
        index_list = [0, 1, index_of_A[1], index_of_A[1] + 1, 1]

    elif re.match("(A)A*(B)(A*)$", thread_pattern) and  (('you\'re') in tweet_text.lower()
                                                         or ('you are') in tweet_text.lower()
                                                         or ('are you') in tweet_text.lower()
                                                         or ('you were') in tweet_text.lower()
                                                         or ('were you') in tweet_text.lower()):

        index_of_B = ([pos for pos, char in enumerate(thread_pattern) if char == 'B'])

        # putting the position of type of tweet in the index list that is returned
        # cue, oblivious, sarcastic, elicit, person class (1, 2 or 3)
        index_list = [0, -1, index_of_B[0], index_of_B[0] + 1, 2]

    elif re.match("(A)(A*B[AB]*)(C)([AB]*)$", thread_pattern) and (('she is') in tweet_text.lower()
                                                                   or ('she was') in tweet_text.lower()
                                                                   or ('was she') in tweet_text.lower()
                                                                   or ('she\'s') in tweet_text.lower()
                                                                   or ('he is') in tweet_text.lower()
                                                                   or ('he was') in tweet_text.lower()
                                                                   or ('was he') in tweet_text.lower()
                                                                   or ('he\'s') in tweet_text.lower()
                                                                   or ('they were') in tweet_text.lower()
                                                                   or ('they\'re') in tweet_text.lower()
                                                                   or ('were they') in tweet_text.lower()
                                                                   or ('they are') in tweet_text.lower()):

        index_of_C = ([pos for pos, char in enumerate(thread_pattern) if char == 'C'])

        #putting the position of type of tweet in the index list that is returned
        #cue, oblivious, sarcastic, elicit, person class (1, 2 or 3)
        index_list = [0, 1, index_of_C[0], index_of_C[0] + 1, 3]

    return index_list


#main function, calling all the others and writing into the dataset
def define_expressions_for_tweet():

    df = pd.read_csv(filtered)


    new_dataset = pd.DataFrame([], columns=["pattern", "person", "cue_id", "sar_id", "obl_id", "eli_id",
                                            "perspective", "cue_text", "sar_text", "obl_text", "eli_text",
                                            "cue_user", "sar_user", "obl_user", "eli_user"])
    raw_dataset = pd.DataFrame([], columns=["pattern", "raw_json_replies"])

    index = new_dataset.shape[0]
    raw_index = raw_dataset.shape[0]
    for i in range(0, df.shape[0]):

        replies_list = get_thread_recursively(df.at[i,'tweet_id'])
        pattern = ''
        author_id_list = {}
        letter = 'A'


        for x in replies_list:
            current_id = x[2]
            if current_id not in author_id_list:
                author_id_list[current_id] = letter
                pattern += letter
                letter = next_alpha(letter)
            else:
                pattern += author_id_list[current_id]


        raw_dataset.loc[raw_index, 'pattern'] = pattern
        raw_dataset.loc[raw_index, 'raw_json_replies'] = replies_list
        raw_index = raw_index + 1
        raw_dataset.to_csv('raw_dataset.csv', index=False)

        index_list_types = recognize_type_of_tweet(pattern, replies_list)

        print(i, index_list_types, pattern, replies_list)

        if len(index_list_types) < 4 or len(pattern) < 1:
            continue

        try:
            new_dataset.loc[index, "pattern"] = pattern

            if index_list_types[4] == 1:
                new_dataset.loc[index, "person"] = '1ST'
                new_dataset.loc[index, "perspective"] = "INTENDED"
            elif index_list_types[4] == 2:
                new_dataset.loc[index, "person"] = '2ND'
                new_dataset.loc[index, "perspective"] = "PERCEIVED"
            elif index_list_types[4] == 3:
                new_dataset.loc[index, "person"] = '3RD'
                new_dataset.loc[index, "perspective"] = "PERCEIVED"

            # cue, oblivious, sarcastic, elicit
            cue = replies_list[index_list_types[0]]
            new_dataset.loc[index, "cue_id"] = cue[0]
            new_dataset.loc[index, "cue_text"] = cue[1]
            new_dataset.loc[index, "cue_user"] = cue[2]

            if index_list_types[1] != -1:
                obl = replies_list[index_list_types[1]]
                new_dataset.loc[index, "obl_id"] = obl[0]
                new_dataset.loc[index, "obl_text"] = obl[1]
                new_dataset.loc[index, "obl_user"] = obl[2]

            sar = replies_list[index_list_types[2]]
            new_dataset.loc[index, "sar_id"] = sar[0]
            new_dataset.loc[index, "sar_text"] = sar[1]
            new_dataset.loc[index, "sar_user"] = sar[2]

            if index_list_types[3] < len(replies_list):
                eli = replies_list[index_list_types[3]]
                new_dataset.loc[index, "eli_id"] = eli[0]
                new_dataset.loc[index, "eli_text"] = eli[1]
                new_dataset.loc[index, "eli_user"] = eli[2]

            index = index + 1
            new_dataset.to_csv("new_dataset.csv", index=False)


        except IndexError:
            _ = 0

    return

define_expressions_for_tweet()








