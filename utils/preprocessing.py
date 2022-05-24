import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import emoji
from gensim.parsing.preprocessing import remove_stopwords

def remove_na_from_column(df, column_name):
    df = df.dropna(subset = [column_name])
    df = df.reset_index(drop = True)

    return df

EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def preprocess_tweets(df, column_name='tweet'):
    df[column_name] = df[column_name].transform(process_tweet)

    return df

def process_tweet(s, keep_emoji=False, keep_usernames=False):

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
    s = emoji.demojize(s)

    if keep_emoji:
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

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