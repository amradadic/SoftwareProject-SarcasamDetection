import numpy as np
import json
from os import path
import math
from sklearn import metrics as m
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from utils import preprocessing

# makes dataframe and does preprocessing
def preparing_data(df_SPIRS_non_sarcastic, df_SPIRS_sarcastic):
    non_sarcastic_tweets = np.array(df_SPIRS_non_sarcastic['sar_text'])
    non_sarcastic_tweet_id = np.array(df_SPIRS_non_sarcastic['sar_id'])
    label = np.zeros(len(non_sarcastic_tweets), dtype=np.int8)
    dataset_nonsarcasm = pd.DataFrame(
        {'tweet_id': list(non_sarcastic_tweet_id), 'label': label, 'tweet': list(non_sarcastic_tweets)},
        columns=['tweet_id', 'label', 'tweet'])

    sarcastic_tweets = np.array(df_SPIRS_sarcastic['sar_text'])
    sarcastic_tweets_id = np.array(df_SPIRS_sarcastic['sar_id'])
    label = np.ones(len(sarcastic_tweets), dtype=np.int8)
    dataset_sarcasm = pd.DataFrame(
        {'tweet_id': list(sarcastic_tweets_id), 'label': label, 'tweet': list(sarcastic_tweets)},
        columns=['tweet_id', 'label', 'tweet'])

    df_SPIRS = pd.concat([dataset_nonsarcasm, dataset_sarcasm], ignore_index=True)

    df_SPIRS = preprocessing.remove_na_from_column(df_SPIRS, 'tweet')
    df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS)

    return df_SPIRS


def train_test_split(df, ratio=0.8):
    df_SPIRS = df.sample(frac=1)

    y = df_SPIRS['label'].values
    X = df_SPIRS['tweet'].values

    n_train = math.floor(ratio * X.shape[0])
    # n_test = math.ceil((1-0.8) * X.shape[0])
    x_train = X[:n_train]
    y_train = y[:n_train]
    x_test = X[n_train:]
    y_test = y[n_train:]

    tweet_id = df_SPIRS['tweet_id'][n_train:].values

    # (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=123, shuffle=True)

    return (x_train, x_test, y_train, y_test, tweet_id)


def metrics(y_test, y_pred, target_names):
    tn, fp, fn, tp = m.confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    dict_confusion = {'True negative' : int(tn),
          'False positive' : int(fp),
          'False negative' : int(fn),
          'True positive' : int(tp),
          }
    dict_report = m.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return dict_confusion | dict_report


def json_predictions(file_name, prediction_model, converting_method, metrics, df):
    dictionary = {'Model': prediction_model,
                  'Method for converting text data': converting_method,
                  'Metrics': metrics,
                  'Data': df.to_dict('records')}

    if path.isfile(file_name): #file exist
        with open(file_name) as fp:
            listObj = json.load(fp)

        listObj.append(dictionary)

        with open(file_name, 'w') as json_file:
            json.dump(listObj, json_file, indent=4)
    else:
        with open(file_name, 'w') as json_file:
            json.dump([dictionary], json_file, indent=4)
