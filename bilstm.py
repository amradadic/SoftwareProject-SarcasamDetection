#%%
##BiLSTM KERAS
import numpy as np
import pandas as pd
from sympy import Predicate
from utils import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from utils import preprocessing

#%%
n_unique_words = 10000
maxlen = 50

#%%

df_SPIRS_sarcastic = pd.read_csv('SPIRS-sarcastic.csv')
df_SPIRS_non_sarcastic = pd.read_csv('SPIRS-non-sarcastic.csv')

#remove NAs from sar_text
df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_text')

#fill na from other columns
df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.fill_na_from_column(df_SPIRS_sarcastic, 'cue_text')
#non sar has no cue text

#preprocess columns
df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'sar_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'eli_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'eli_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'obl_text')
df_SPIRS_non_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_non_sarcastic, 'obl_text')

df_SPIRS_sarcastic = preprocessing.preprocess_tweets(df_SPIRS_sarcastic, 'cue_text')
#non sar has no cue text

#without context
#df_SPIRS_sarcastic = df_SPIRS_sarcastic[['sar_text']]
#df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic[['sar_text']]

#with context
#get context
df_SPIRS_sarcastic = preprocessing.get_df_context(df_SPIRS_sarcastic, cue = True)
df_SPIRS_non_sarcastic = preprocessing.get_df_context(df_SPIRS_non_sarcastic, cue = True)

#add labels
df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

#concat
df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

#test train split
x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, ~df_SPIRS.columns.isin(['sar_id', 'label'])], df_SPIRS[['label']], test_size=0.2, random_state=123, shuffle=True)

#%%

tokenizer, dictionary = preprocessing.get_dictionary(x_train, n_unique_words)
embedding = preprocessing.get_glove_embedding_BiLSTM(dictionary)
    

# %%
#embedding_layer = my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
# %%

def BiLSTM(dict_len, embedding, max_len):
    
    model_glove = Sequential()
    model_glove.add(Embedding(input_dim=dict_len, output_dim=len(embedding[0]), input_length=max_len, weights=[embedding], trainable=True))
    model_glove.add(Bidirectional(LSTM(20, return_sequences=True)))
    model_glove.add(Dropout(0.2))
    model_glove.add(BatchNormalization())
    model_glove.add(Bidirectional(LSTM(20, return_sequences=True)))
    model_glove.add(Dropout(0.2))
    model_glove.add(BatchNormalization())
    model_glove.add(Bidirectional(LSTM(20)))
    model_glove.add(Dropout(0.2))
    model_glove.add(BatchNormalization())
    model_glove.add(Dense(64, activation='relu'))
    model_glove.add(Dense(64, activation='relu'))
    model_glove.add(Dense(1, activation='sigmoid'))
    model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_glove

model = BiLSTM(len(dictionary), embedding, maxlen) 


#%%
#converting the tweets to their index form by using the texts_to_sequences 
#padding the sequences so all of them have the same length

#no context
#X_train_indices = tokenizer.texts_to_sequences(x_train['tweets'])
#X_train_indices = pad_sequences(X_train_indices, maxlen=maxlen, padding='post')

#X_test_indices = tokenizer.texts_to_sequences(x_test['tweets'])
#X_test_indices = pad_sequences(X_test_indices, maxlen=maxlen, padding='post')

#context
X_train_indices = tokenizer.texts_to_sequences(preprocessing.concat_df(x_train)['tweets'])
X_train_indices = pad_sequences(X_train_indices, maxlen=maxlen, padding='post')

X_test_indices = tokenizer.texts_to_sequences(preprocessing.concat_df(x_test)['tweets'])
X_test_indices = pad_sequences(X_test_indices, maxlen=maxlen, padding='post')
 
#%%
model.fit(X_train_indices, y_train, epochs = 5)
# %%
y_pred = model.predict(X_test_indices)

for i in  range(0, len(y_pred)):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else: 
        y_pred[i] = 0
        
# %%

print(classification_report(y_test['label'], y_pred, target_names=['Non-sarcastic', 'Sarcastic']))

# %%
