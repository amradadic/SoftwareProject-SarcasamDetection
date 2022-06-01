import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import emoji
from gensim.parsing.preprocessing import remove_stopwords
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
string.punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import glove


def remove_na_from_column(df, column_name):
    df = df.dropna(subset = [column_name])
    df = df.reset_index(drop = True)

    return df

def fill_na_from_column(df, column_name):
    df[column_name] = df[column_name].fillna('')

    return df

EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def preprocess_tweets(df, column_name='tweet', keep_emoji = True):
    df[column_name] = df[column_name].transform(func = process_tweet, keep_emoji=keep_emoji, keep_usernames=False)

    return df

def process_tweet(s, keep_emoji=True, keep_usernames=False):

    s = s.lower()

    #removing urls, htmls tags, etc
    s = re.sub(r'https\S+', r'', str(s))
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    #removing stopwords
    s = remove_stopwords(s)

    #removing emojis
    if keep_emoji:
        s = emoji.demojize(s)
    else:
        emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

        s = emoj.sub(r'',s)

 #   s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)

    #removing hashtags
    s = re.sub(HASHTAG_BEFORE, r'\1!!', s)


    #removing usernames

    #removing just @sign
    if keep_usernames:
        s = ' '.join(s.split())

        s = re.sub(LEADING_NAMES, r' ', s)
        s = re.sub(TAIL_NAMES, r' ', s)

        s = re.sub(FIND_MENTIONS, r'\1', s)

    #removing username completely
    else:
        s = re.sub(FIND_MENTIONS, r' ', s)
    
    #removing username tags - just in case ??
    s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    user_regex = r".?@.+?( |$)|<@mention>"    
    s = re.sub(user_regex," @user ", s, flags=re.I)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())

    return s

def remove_punctiation(df, column_name='tweet'):
    df[column_name] = df[column_name].transform(remove_punctuation)

    return df
    
def remove_punctuation(text):
    if(type(text)==float):
        return text
    
    ans=""  
    for i in text:     
        if i not in string.punctuation:
            ans+=i    
            
    return ans

def remove_nltk_stopwords(df, column_name='tweet') :
    df[column_name] = df[column_name].transform(remove_nltk_stopwords_from_tweet)

    return df
    
    
def remove_nltk_stopwords_from_tweet(s):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(s)
    tokens_without_sw = [word for word in word_tokens if not word in stop_words]
    
    s = (" ").join(tokens_without_sw)
    
    return s


#returns dataframe with columns sar_text, 'obl_text', 'elicit text' [and 'cue text'] and corrensponding id
def get_df_context(df, cue = False) :
    
    if cue:
        if 'cue_text' in df.columns:
            df = df[['sar_id', 'sar_text', 'obl_text', 'eli_text', 'cue_text']]
        else:
            df = df[['sar_id', 'sar_text', 'obl_text', 'eli_text']]
            df = df.assign(cue_text='')
        
    else:
        df = df[['sar_id', 'sar_text', 'obl_text', 'eli_text']]
        
    return df

def vectorize(xs, vectorizer=TfidfVectorizer(min_df=1, stop_words="english")):
    text = [' '.join(x) for x in xs]
    return vectorizer.fit_transform(text)
    

def get_tfidf_context(df_train, df_test) :
    
    ncol = df_train.shape[1]
 
    tfidf = TfidfVectorizer()
    tfidf.fit(df_train['sar_text'])
    
    tfidf_train_sar = tfidf.transform(df_train['sar_text'])
    tfidf_train_eli = tfidf.transform(df_train['eli_text'])
    tfidf_train_obl = tfidf.transform(df_train['obl_text'])

    tfidf_test_sar = tfidf.transform(df_test['sar_text'])
    tfidf_test_eli = tfidf.transform(df_test['eli_text'])
    tfidf_test_obl = tfidf.transform(df_test['obl_text'])
    
    #cue = True
    if ncol == 4:
        tfidf_train_cue = tfidf.transform(df_train['cue_text'])
        tfidf_test_cue = tfidf.transform(df_test['cue_text'])
    
    #final tfidf vectors
    if ncol == 3:
        tfidf_train = (tfidf_train_sar + tfidf_train_eli + tfidf_train_obl) / 3
        tfidf_test = (tfidf_test_sar + tfidf_test_eli + tfidf_test_obl) / 3
    elif ncol == 4:
        tfidf_train = (tfidf_train_sar + tfidf_train_eli + tfidf_train_obl + tfidf_train_cue) / 4
        tfidf_test = (tfidf_test_sar + tfidf_test_eli + tfidf_test_obl + tfidf_test_cue) / 4

    
    return tfidf_train, tfidf_test



def get_glove_embedding_SVM(df_train, df_test):
    model = glove.load_glove()
    
    ncol = df_train.shape[1]
    
    print('ncol' + str(ncol))
    
    # Set a word vectorizer
    vectorizer = glove.GloveVectorizer(model)
    print('sarcastic')
    # Get the sentence embeddings for the train dataset
    Xtrain_sar = vectorizer.fit_transform(df_train['sar_text'])
    # Get the sentence embeddings for the test dataset
    Xtest_sar = vectorizer.transform(df_test['sar_text'])
    
    
    if ncol >= 3:
        print("elicit - 10735 NaN values in train, 2709 NaN values in test")
        Xtrain_eli = vectorizer.transform(df_train['eli_text'])
        Xtest_eli = vectorizer.transform(df_test['eli_text'])
        print('oblivious - 8889 NaN values in train, 2252 NaN values in test')
        Xtrain_obl = vectorizer.transform(df_train['obl_text'])
        Xtest_obl = vectorizer.transform(df_test['obl_text'])
        
    if ncol == 4:
        print('cue')
        print('oblivious - 9317 NaN values in train, 2335 NaN values in test')
        Xtrain_cue = vectorizer.transform(df_train['cue_text'])
        Xtest_cue= vectorizer.transform(df_test['cue_text'])
        
        
    #final glove vectors
    if ncol == 1:
        Xtrain = Xtrain_sar
        Xtest = Xtest_sar
    if ncol == 3:
        Xtrain = (Xtrain_sar + Xtrain_eli + Xtrain_obl) / 3
        Xtest = (Xtest_sar + Xtest_eli + Xtest_obl) / 3
    elif ncol == 4:
        Xtrain = (Xtrain_sar + Xtrain_eli + Xtrain_obl + Xtrain_cue) / 4
        Xtest = (Xtest_sar + Xtest_eli + Xtest_obl + Xtest_cue) / 4

    return Xtrain, Xtest
    
        

    
    