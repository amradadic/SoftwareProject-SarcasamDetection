import pandas as pd
from utils import preprocessing
from sklearn.model_selection import train_test_split

df_SPIRS_sarcastic = pd.read_csv('data/SPIRS-sarcastic.csv')
df_SPIRS_non_sarcastic = pd.read_csv('data/SPIRS-non-sarcastic.csv')

# remove NAs from sar_text
df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_text')

df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'cue_text')

# fill na from other columns
df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'cue_text')
# non sar has no cue text


# preprocess columns
df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'sar_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'cue_text')
# non sar has no cue text

# add labels
df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

# concat
df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, ~df_SPIRS.columns.isin(['label'])], df_SPIRS['label'], test_size=0.2, random_state=123, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=123, shuffle=True)

x_train.to_csv('data/x_train_context_remove_na.csv', index=False)
x_test.to_csv('data/x_test_context_remove_na.csv', index=False)
x_val.to_csv('data/x_val_context_remove_na.csv', index=False)
y_train.to_csv('data/y_train_context_remove_na.csv', index=False)
y_test.to_csv('data/y_test_context_remove_na.csv', index=False)
y_val.to_csv('data/y_val_context_remove_na.csv', index=False)

print(x_train)
'''
x_train = pd.read_csv('data/x_train_context_remove_na.csv')
x_test = pd.read_csv('data/x_test_context_remove_na.csv')
x_val = pd.read_csv('data/x_val_context_remove_na.csv')
y_train = pd.read_csv('data/y_train_context_remove_na.csv')
y_test = pd.read_csv('data/y_test_context_remove_na.csv')
y_val = pd.read_csv('data/y_val_context_remove_na.csv')
print(x_train, x_test, x_val)
print(y_train, y_test, y_val)
'''