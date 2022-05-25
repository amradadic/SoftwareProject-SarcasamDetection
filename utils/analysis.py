import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import plotly.express as px


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


def get_top_text_ngrams(corpus, n, g, mode=1):
    if mode == 2:
        vec = TfidfVectorizer(ngram_range=(g, g)).fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

#mode 2 - TF-ID
def draw_plot_for_common_ngrams(text, n=1, number_of_common=20, name_of_ngram="N-gram", mode=1):
    most_common = get_top_text_ngrams(text, number_of_common, n, 2)
    most_common = dict(most_common)
    temp = pd.DataFrame(columns=["Common_words", "Count"])
    temp["Common_words"] = list(most_common.keys())
    temp["Count"] = list(most_common.values())
    fig = px.bar(temp, x="Count", y="Common_words", title="Common " + name_of_ngram + " in Text", orientation='h', width = 1200,
                color='Common_words', color_discrete_sequence=px.colors.qualitative.Plotly)
    
    fig.layout.showlegend = False
    fig.show()
