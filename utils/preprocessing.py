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

def preprocess_tweets(df, column_name='tweet', keep_emoji = True):
    df[column_name] = df[column_name].transform(func = process_tweet, keep_emoji=keep_emoji)

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