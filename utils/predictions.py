import numpy as np
import json
from os import path
import math
from matplotlib import pyplot as plt
from sklearn import metrics as m
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from utils import preprocessing


# makes dataframe and does preprocessing
def make_dataframe(df_SPIRS_non_sarcastic, df_SPIRS_sarcastic, context = False, cue = False, emoji = True):
    if context:  #retuns dataframe with columns 'sar_id', 'obl_id', 'eli_id', 'cue_id', 'sar_text', 'obl_text', 'eli_text', 'cue_text'
        #remove NA from sar_text
        df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_text')
        df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_text')

        # fill NA from other columns
        df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'eli_text')
        df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'eli_text')

        df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'obl_text')
        df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'obl_text')

        if cue:
            df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'cue_text')
            # non sar has no cue text

        df_SPIRS_sarcastic = preprocessing.get_df_context(df_SPIRS_sarcastic, cue=cue)
        df_SPIRS_non_sarcastic = preprocessing.get_df_context(df_SPIRS_non_sarcastic, cue=cue)

        df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
        df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

        df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

        df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS, 'sar_text', keep_emoji=emoji)
        df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS, 'obl_text', keep_emoji=emoji)
        df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS, 'eli_text', keep_emoji=emoji)
        if cue:
            df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS, 'cue_text', keep_emoji=emoji)

    else:  #returns dataframe with columns tweet_id, tweet, label
        non_sarcastic_tweets = np.array(df_SPIRS_non_sarcastic['sar_text'])
        non_sarcastic_tweet_id = np.array(df_SPIRS_non_sarcastic['sar_id'])
        label = np.zeros(len(non_sarcastic_tweets), dtype=np.int8)
        dataset_nonsarcasm = pd.DataFrame({'tweet_id': list(non_sarcastic_tweet_id), 'label': label, 'tweet': list(non_sarcastic_tweets)},
                                          columns=['tweet_id', 'label', 'tweet'])

        sarcastic_tweets = np.array(df_SPIRS_sarcastic['sar_text'])
        sarcastic_tweets_id = np.array(df_SPIRS_sarcastic['sar_id'])
        label = np.ones(len(sarcastic_tweets), dtype=np.int8)
        dataset_sarcasm = pd.DataFrame({'tweet_id': list(sarcastic_tweets_id), 'label': label, 'tweet': list(sarcastic_tweets)},
                                       columns=['tweet_id', 'label', 'tweet'])
        df_SPIRS = pd.concat([dataset_nonsarcasm, dataset_sarcasm], ignore_index=True)

        df_SPIRS = preprocessing.remove_na_from_column(df_SPIRS, 'tweet')
        df_SPIRS = preprocessing.preprocess_tweets(df_SPIRS)

    return df_SPIRS


def metrics(y_test, y_pred, target_names):
    tn, fp, fn, tp = m.confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    dict_confusion = {'True negative' : int(tn),
          'False positive' : int(fp),
          'False negative' : int(fn),
          'True positive' : int(tp),
          }
    dict_report = m.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return dict_confusion | dict_report


def json_metrics(file_name, prediction_model, converting_method, metrics, df):
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


def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    plt.figure(figsize=(10, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()
