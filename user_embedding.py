import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from utils import preprocessing

def user_representaton_tfidf(tweet, user_tweets, num_token = 512):
    tfidf = TfidfVectorizer()
    try:
        tweet_tokenizer = tfidf.fit_transform(tweet)
        tweet_tokens = tfidf.get_feature_names_out()
    except:
        tweet_tokens = []

    tfidf = TfidfVectorizer()
    tokenization = tfidf.fit_transform(user_tweets)
    tokens = tfidf.get_feature_names_out()
    try:
        random_tokens = random.sample(list(tokens), num_token)
    except:
        random_tokens = random.sample(list(tokens), len(tokens))

    return np.concatenate([tweet_tokens, random_tokens[:num_token-len(tweet_tokens)]])


def user_embeddings(df_SPIRS, history):
   tweet = df_SPIRS['sar_text']
   users = df_SPIRS['sar_user']
   embeddings = {}
   for i in range(100): #(len(tweet)):
       user = users[i].split('|')
       user_id = int(user[-1])

       #find tweets of user
       tweets_df = history[history['sar_user'] == user_id]
       user_tweets = tweets_df['sar_text'].values

       if len(user_tweets) > 0:
           embeddings[user_id] = user_representaton_tfidf([tweet[i]], user_tweets)

   return embeddings

history_sar = pd.read_csv('SPIRS-sarcastic-history2.txt', names=['sar_user', 'sar_id', 'sar_text'], sep='\t')
history_non_sar = pd.read_csv('SPIRS-non-sarcastic-history2.txt', names=['sar_user', 'sar_id', 'sar_text'], sep='\t')
history = pd.concat([history_sar, history_non_sar], ignore_index=True)
history = preprocessing.preprocess_tweets(history, 'sar_text')

df_SPIRS_sarcastic = pd.read_csv('data\SPIRS-sarcastic2.csv')
df_SPIRS_non_sarcastic = pd.read_csv('data\SPIRS-non-sarcastic2.csv')
df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic])
df_SPIRS = pd.DataFrame({'sar_user': np.array(df_SPIRS['sar_user']), 'sar_text': np.array(df_SPIRS['sar_text'])})
df_SPIRS = preprocessing.remove_na_from_column(df_SPIRS, 'sar_user')
df_SPIRS = preprocessing.remove_na_from_column(df_SPIRS, 'sar_text')
df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS, 'sar_text')


embeddings = user_embeddings(df_SPIRS, history)
print(embeddings)
out_file = 'user_embeddings_tfidf.txt'
with open(out_file, 'w') as f:
    for user, embedding in embeddings.items():
        temp = str(user)

        for val in embedding:
            temp += ' ' + val

        f.write(temp + '\n')