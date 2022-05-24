import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud

def n_sar_nonsar_tweets(df):
    sarcastic_count = len(df[df.label==1])
    nonsarcastic_count = len(df[df.label==0])
    print("SPIRS - sarcastic tweet count: " + str(sarcastic_count))
    print("SPIRS - non sarcastic tweet count: " + str(nonsarcastic_count))

def tweet_length_graph(df):
    df_sarcastic = df[df.label==1]
    df_nonsarcastic = df[df.label==0]

    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Average tweet length", fontsize=14)
    axes[0].set_title('Sarcastic tweets')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Number of tweets')
    axes[1].set_title('Nonsarcastic tweets')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Number of tweets')
    
    df_sarcastic['tweet'][1:2000].str.len().hist(ax=axes[0], bins=10, range=(0,600), color='pink')
    df_nonsarcastic['tweet'][1:2000].str.len().hist(bins=10, ax=axes[1], range=(0,600))
    plt.show() 


def word_length_graph(df):
    df_sarcastic = df[df.label==1]
    df_nonsarcastic = df[df.label==0]
    
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Average word length", fontsize=14)
    axes[0].set_title('Sarcastic tweets')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Number of words')
    axes[1].set_title('Nonsarcastic tweets')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Number of words')

    df_sarcastic['tweet'][1:2000].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist(ax=axes[0], range=(0,60), color = 'pink')
    df_nonsarcastic['tweet'][1:2000].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist(ax=axes[1], range=(0,60))

    plt.show()

def wordcloud(df):
    #uporredit rezultat s dzenetinim

    text = " ".join(tweet for tweet in df.tweet)

    wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)

    wordcloud=wordcloud.generate(text)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()