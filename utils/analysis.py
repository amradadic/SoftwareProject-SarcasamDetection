import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from wordcloud import WordCloud

def n_sar_nonsar_tweets(df):
    sarcastic_count = len(df[df.sarcasm_label=="sarcastic"])
    nonsarcastic_count = len(df[df.sarcasm_label=="not_sarcastic"])
    print("SPIRS - sarcastic tweet count: " + str(sarcastic_count))
    print("SPIRS - non sarcastic tweet count: " + str(nonsarcastic_count))

def tweet_length_graph(df):
    df_sarcastic = df[df.sarcasm_label=="sarcastic"]
    df_nonsarcastic = df[df.sarcasm_label=="not_sarcastic"]

    df_sarcastic['sar_text'].str.len().hist(label='Sarcastic tweets')
    df_nonsarcastic['sar_text'].str.len().hist(color='Nonsarcastic tweets')

def word_length_graph(df):
    df_sarcastic = df[df.sarcasm_label=="sarcastic"]
    df_nonsarcastic = df[df.sarcasm_label=="not_sarcastic"]

    df_sarcastic['sar_text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
    df_nonsarcastic['sar_text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()

def wordcloud(df):
    #uporredit rezultat s dzenetinim
    stopwords=set(stopwords.words('english'))

    text = " ".join(tweet for tweet in df_SPIRS_sarcastic.sar_text)

    wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)

    wordcloud=wordcloud.generate(text)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()