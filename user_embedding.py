import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from utils import preprocessing
import zipfile
import csv

def user_representaton_tfidf(tweet, user_tweets, num_token = 512):
    tfidf = TfidfVectorizer()
    try:
        tweet_tokenizer = tfidf.fit_transform(tweet)
        tweet_tokens = tfidf.get_feature_names_out()
    except:
        tweet_tokens = []

    tfidf = TfidfVectorizer()

    try:
        tokenization = tfidf.fit_transform(user_tweets)
        tokens = tfidf.get_feature_names_out()
        try:
            random_tokens = random.sample(list(tokens), num_token)
        except:
            random_tokens = random.sample(list(tokens), len(tokens))
        return np.concatenate([tweet_tokens, random_tokens[:num_token - len(tweet_tokens)]])
    except:
        raise ValueError


def user_embeddings(df_SPIRS, history):
   tweet = df_SPIRS['sar_text']
   users = df_SPIRS['sar_user']
   embeddings = {}

   for i in range(len(tweet)):
       user = users[i].split('|')
       user_id = int(user[-1])

       #find tweets of user
       tweets_df = history[history['sar_user'] == user_id]
       user_tweets = tweets_df['sar_text'].values

       if len(user_tweets) > 0:
           try:
              embeddings[user_id] = user_representaton_tfidf([tweet[i]], user_tweets)
           except ValueError:
               continue

   return embeddings


history_sar = []
history_non_sar = []
with zipfile.ZipFile('data/spirs_history.zip') as zip:
  a = 0
  file = zip.open('spirs_history/SPIRS-sarcastic-history.txt', mode='r')
  for chunk in pd.read_csv(file, names=['sar_user', 'sar_id', 'sar_text'], sep='\t', chunksize=1024):
    history_sar.append(preprocessing.preprocess_tweets(chunk, 'sar_text'))
    a += 1
    if a == 1000:
        break

history_sarcastic = pd.concat(history_sar, ignore_index=True)

df_SPIRS_sarcastic = pd.read_csv('data\SPIRS-sarcastic2.csv')
df_SPIRS_sarcastic = pd.DataFrame({'sar_user': np.array(df_SPIRS_sarcastic['sar_user']), 'sar_text': np.array(df_SPIRS_sarcastic['sar_text'])})
df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_user')
df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'sar_text')

embeddings_sar = user_embeddings(df_SPIRS_sarcastic, history_sarcastic)
print(embeddings_sar)
out_file = 'user_embeddings_sar_tfidf.csv'
with open(out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['user_id', 'token', 'label']
    writer.writerow(header)
    for user, embedding in embeddings_sar.items():
        token = ''
        for val in embedding:
            token += val + ' '
        row = [user, token, 1]
        writer.writerow(row)



with zipfile.ZipFile('data/spirs_history.zip') as zip:
  a = 0
  file = zip.open('spirs_history/SPIRS-non-sarcastic-history.txt', mode='r')
  for chunk in pd.read_csv(file, names=['sar_user', 'sar_id', 'sar_text'], sep='\t', chunksize=1024):
    history_non_sar.append(preprocessing.preprocess_tweets(chunk, 'sar_text'))
    a += 1
    if a == 1000:
        break

history_non_sarcastic = pd.concat(history_non_sar, ignore_index=True)

df_SPIRS_non_sarcastic = pd.read_csv('data\SPIRS-non-sarcastic2.csv')
df_SPIRS_non_sarcastic = pd.DataFrame({'sar_user': np.array(df_SPIRS_non_sarcastic['sar_user']), 'sar_text': np.array(df_SPIRS_non_sarcastic['sar_text'])})
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_user')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'sar_text')

embeddings_non_sar = user_embeddings(df_SPIRS_non_sarcastic, history_non_sarcastic)
out_file = 'user_embeddings_non_sar_tfidf.csv'
with open(out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['user_id', 'token', 'label']
    writer.writerow(header)
    for user, embedding in embeddings_non_sar.items():
        token = ''
        for val in embedding:
            token += val + ' '
        row = [user, token, 0]
        writer.writerow(row)
