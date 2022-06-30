import requests
import os
import json

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAI3qcQEAAAAA4PAEfL%2ByQCQVQJpHQdNdCoBtlS0%3DvNGdgosAW0lrgv5t8alZ2buQSRfdIYVNT4KCw1Xyvql09qulAS'


#curl --request GET \
#  --url 'https://api.twitter.com/2/tweets/search/recent?query=conversation_id:1279940000004973111&
#  tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id' \
#  --header 'Authorization: Bearer $BEARER_TOKEN'

search_url = "https://api.twitter.com/2/tweets/search/recent"
#search_url = "https://api.twitter.com/2/tweets"
# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields


#pokusaj amra
ids = '1540746631515103232 -is:retweet -is:quote'
expansions = 'author_id,in_reply_to_user_id,referenced_tweets.id'
#fields = 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'

userfields = 'name,username'
#query_params = {'ids' : ids, 'tweet.fields' :fields, 'expansions' : expansions, 'user.fields': userfields}

#pokusaj esma
expansions = 'referenced_tweets.id,in_reply_to_user_id'
fields = 'in_reply_to_user_id,author_id,created_at,conversation_id'

query_params = {'query': ids, 'tweet.fields' :fields, 'expansions' : expansions}

#
#userfields = 'name,username'
#query_params = {'ids' : ids, 'tweet.fields' :fields, 'expansions' : expansions, 'user.fields': userfields}



def getConversationId(tweet_id):
    search_url = "https://api.twitter.com/2/tweets"

    expansions = 'author_id,in_reply_to_user_id,referenced_tweets.id'
    fields = 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'
    userfields = 'name,username'
    query_params = {'ids' : tweet_id, 'tweet.fields' :fields, 'expansions' : expansions, 'user.fields': userfields}
    json_response = connect_to_endpoint(search_url, query_params)
    print(json.dumps(json_response, indent=4, sort_keys=True))

    return json_response['data'][0]['conversation_id']


def getConversationTweets(conversation_id):    #ove stvari rade u fajlu twitter API vjerovatno je neka glupost pvdje zaboravljena pa nece
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    expansions = 'referenced_tweets.id,in_reply_to_user_id'
    fields = 'in_reply_to_user_id,author_id,created_at,conversation_id'

    query_params = {'query': conversation_id, 'tweet.fields': fields, 'expansions': expansions, 'max_results': 100}
    json_response = connect_to_endpoint(search_url, query_params)
    print(json.dumps(json_response, indent=4, sort_keys=True))
    #this should filter the json, remove the rt and other extras and return the conversation as a list
    #print(json_response['data'][0]['text'])
    #print(json_response['data'][0]['referenced_tweets'][0]['type']) #this needs to be replied_to
    #   -is:quote-is:retweet -is:quote




def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params =params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    json_response = connect_to_endpoint(search_url, query_params)
    print(json.dumps(json_response, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()