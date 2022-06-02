import GetOldTweets3 as got


def extract_tweets(hashtag):
    gettweet = got.manager.TweetCriteria().setQuerySearch(hashtag) \
        .setSince("2020-01-01") \
        .setUntil("2020-02-01") \
        .setMaxTweets(10)

    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(gettweet)

    # Creating list of chosen tweet data
    text_tweets = [[tweet.text] for tweet in tweets]
    print(text_tweets)


# calling the function
extract_tweets('being sarcastic')


