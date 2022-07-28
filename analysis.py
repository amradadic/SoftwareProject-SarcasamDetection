#%%
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

#%%
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

#%%
#removing rows with missing tweets
df_SPIRS = preprocessing.remove_na_from_column(df_SPIRS, 'tweet')

#preprocessing tweets
df_SPIRS_no_emojis = df_SPIRS.copy()
df_SPIRS_no_emojis = preprocessing.preprocess_tweets(df_SPIRS_no_emojis, column_name='tweet', keep_emoji=False)

df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS, column_name='tweet')
print(df_SPIRS.head())


#%%

##ANALYSIS

#number of sarcastic/non sarcastic tweets
analysis.n_sar_nonsar_tweets(df_SPIRS)

#average tweet length graphs - first 2000 sarcastic and 2000 nonsarcastic tweets
analysis.tweet_length_graph(df_SPIRS)

#average word length graphs - first 2000 sarcastic and 2000 nonsarcastic tweets
analysis.word_length_graph(df_SPIRS)

#wordcloud of sarcastic tweets with emojis
analysis.wordcloud(df_SPIRS[df_SPIRS.label==1])

#wordcloud of sarcastic tweets without emojis
analysis.wordcloud(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==1])


#for ngrams - remove punctiation and nltk stopwords
#df_SPIRS = preprocessing.r(df_SPIRS)
#df_SPIRS = preprocessing.remove_nltk_stopwords(df_SPIRS)
#df_SPIRS_no_emojis = preprocessing.remove_nltk_stopwords(df_SPIRS_no_emojis)

#%%
#ngrams of sarcastic with emojis
analysis.draw_plot_for_common_ngrams(df_SPIRS[df_SPIRS.label==1]['tweet'], 1, 20, "Common Unigrams in Text (sarcastic, with emojis)", 1)
#analysis.draw_plot_for_common_ngrams(df_SPIRS[df_SPIRS.label==1]['tweet'], 2, 20, "Common Bigrams in Text (sarcastic, with emojis)", 2)
analysis.draw_plot_for_common_ngrams(df_SPIRS[df_SPIRS.label==1]['tweet'], 3, 20, "Common Trigrams in Text (sarcastic, with emojis)", 1)

#ngrams of sarcastic without emojis
analysis.draw_plot_for_common_ngrams(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==1]['tweet'], 1, 20, "Common Unigrams in Text (sarcastic, without emojis)",1)
#analysis.draw_plot_for_common_ngrams(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==1]['tweet'], 2, 20, "Common Bigrams in Text (sarcastic, without emojis)", 2)
analysis.draw_plot_for_common_ngrams(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==1]['tweet'], 3, 20, "Common Trigrams in Text (sarcastic, without emojis)", 1)

#ngrams of nonsarcastic with emojis
analysis.draw_plot_for_common_ngrams(df_SPIRS[df_SPIRS.label==0]['tweet'], 1, 20, "Common Unigrams in Text (nonsarcastic, with emojis)", 1)
#analysis.draw_plot_for_common_ngrams(df_SPIRS[df_SPIRS.label==0]['tweet'], 2, 20, "Common Bigrams in Text (nonsarcastic, with emojis)", 2)
analysis.draw_plot_for_common_ngrams(df_SPIRS[df_SPIRS.label==0]['tweet'], 3, 20, "Common Trigrams in Text (nonsarcastic, with emojis)", 1)

#ngrams of nonsarcastic without emojis
analysis.draw_plot_for_common_ngrams(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==0]['tweet'], 1, 20, "Common Unigrams in Text (nonsarcastic, without emojis)",1)
#analysis.draw_plot_for_common_ngrams(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==0]['tweet'], 2, 20, "Common Bigrams in Text (nonsarcastic, without emojis)", 2)
analysis.draw_plot_for_common_ngrams(df_SPIRS_no_emojis[df_SPIRS_no_emojis.label==0]['tweet'], 3, 20, "Common Trigrams in Text (nonsarcastic, without emojis)", 1)


# %%
