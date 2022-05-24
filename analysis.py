import pandas as pd
import numpy as np
from utils import analysis
from utils import preprocessing

df_SPIRS_sarcastic = pd.read_csv('SPIRS-sarcastic.csv')
df_SPIRS_non_sarcastic = pd.read_csv('SPIRS-non-sarcastic.csv')

#info
print("\nSarcastic tweets df info:")
print(df_SPIRS_sarcastic.info())
print("\nNon-sarcastic tweets df info:")
print(df_SPIRS_non_sarcastic.info())


##PREPROCESSING

#make new dataframe - only tweets and labels, both sarcastic and nonsarcastic tweets
non_sarcastic_tweets = np.array(df_SPIRS_non_sarcastic['sar_text'])
label = np.zeros(len(non_sarcastic_tweets), dtype = np.int8)
dataset_nonsarcasm = pd.DataFrame({'label': label, 'tweet': list(non_sarcastic_tweets)}, columns=['label', 'tweet'])

sarcastic_tweets = np.array(df_SPIRS_sarcastic['sar_text'])
label = np.ones(len(sarcastic_tweets), dtype = np.int8)
dataset_sarcasm = pd.DataFrame({'label': label, 'tweet': list(sarcastic_tweets)}, columns=['label', 'tweet'])

df_SPIRS = pd.concat([dataset_nonsarcasm, dataset_sarcasm], ignore_index=True)
print(df_SPIRS.head())
print('\n\n\n')

#removing rows with missing tweets
df_SPIRS = preprocessing.remove_na_from_column(df_SPIRS, 'tweet')

#preprocessing tweets
df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS)
print(df_SPIRS.head())



##ANALYSIS

#number of sarcastic/non sarcastic tweets
analysis.n_sar_nonsar_tweets(df_SPIRS)

#average tweet length graphs - first 2000 sarcastic and 2000 nonsarcastic tweets
analysis.tweet_length_graph(df_SPIRS)

#average word length graphs - first 2000 sarcastic and 2000 nonsarcastic tweets
analysis.word_length_graph(df_SPIRS)

#wordcloud of sarcastic tweets
analysis.wordcloud(df_SPIRS[df_SPIRS.label==1])


# %%
