import requests
import os
import json

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAI3qcQEAAAAA4PAEfL%2ByQCQVQJpHQdNdCoBtlS0%3DvNGdgosAW0lrgv5t8alZ2buQSRfdIYVNT4KCw1Xyvql09qulAS'

#search_url = "https://api.twitter.com/2/tweets/search/recent"
search_url = "https://api.twitter.com/2/tweets"
# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
ids = '1225917697675886593'
expansions = 'author_id,in_reply_to_user_id,referenced_tweets.id'
fields = 'author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets'
userfields = 'name,username'
query_params = {'ids' : ids, 'tweet.fields' :fields, 'expansions' : expansions, 'user.fields': userfields}
def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

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