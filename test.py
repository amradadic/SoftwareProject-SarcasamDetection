import tweepy


API_key = "vlY9i6EAhbLvsXkN20CaDjImR"
API_secret = "y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ"
Access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
Access_token_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"

auth = tweepy.OAuthHandler(API_key, API_secret)
auth.set_access_token(Access_token, Access_token_secret)


api = tweepy.API(auth)

tweets_set = set({})
for status in tweepy.Cursor(api.search_tweets, q='star wars', lang='en', until='2022–05–05', tweet_mode='extended').items(10):
   try:
      tweets_set.add((status.retweeted_status.full_text, status.user.screen_name, status.user.followers_count))
   except AttributeError:
      tweets_set.add((status.full_text, status.user.screen_name, status.user.followers_count))

print(len(tweets_set))

searched_tweets = [status for status in tweepy.Cursor(api.search_full_archive,
                                                      label='FullArchive',
                                                      query=querys,
                                                      fromDate='01-01-2022',
                                                      toDate='02-02-2022').items(max_tweets)]

if i.user.id in map:
    map[i.user.id] = map[i.user.id] + 1
map[i.user.id] = 1