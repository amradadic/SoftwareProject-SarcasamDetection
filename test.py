#%%
import pandas as pd
from utils import preprocessing
from sklearn.model_selection import train_test_split
from utils import metrics as pf
from sklearn import svm
from sklearn import metrics as m
from tensorflow.keras.layers import Embedding


##NAPOMENA - KAD SE KORISTI VISE KOLONA NAD SVAKOM KOLONOM TREBA POZVAR .preprocess_tweets

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

#df_SPIRS_sarcastic = preprocessing.get_df_context(df_SPIRS_sarcastic, cue = True)
#df_SPIRS_non_sarcastic = preprocessing.get_df_context(df_SPIRS_non_sarcastic, cue = True)

#df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
#df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

#df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

#x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, df_SPIRS.columns != 'label'], df_SPIRS[['label']], test_size=0.2, random_state=123, shuffle=True)

#### TF-IDF WITH CONTEXT

#tf_idf_train, tf_idf_test = preprocessing.get_tfidf_context(x_train, x_test)

#svm_classifier = svm.LinearSVC().fit(tf_idf_train, y_train['label'])
#tfidf_test_pred = svm_classifier.predict(tf_idf_test)

#pf.plot_coefficients(svm_classifier, tfidf.get_feature_names_out())

#print(m.classification_report(y_test, tfidf_test_pred, target_names=['Non-sarcastic', 'Sarcastic']))


#### GLOVE EMBEDDING

#df_SPIRS_sarcastic = df_SPIRS_sarcastic[['sar_text']]
#df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic[['sar_text']]


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
#get glove embedding
#glove_train, glove_test = preprocessing.get_glove_embedding_SVM(x_train, x_test)
#print(glove_train.shape,  glove_test.shape)

#%%
#SVM
#svm_classifier = svm.LinearSVC().fit(glove_train, y_train['label'])
#glove_test_pred = svm_classifier.predict(glove_test)

#print(m.classification_report(y_test, glove_test_pred, target_names=['Non-sarcastic', 'Sarcastic']))

#%%
#SAMO ZA BiLSTM


#%%

tokenizer, dictionary = preprocessing.get_dictionary(x_train)
embedding = preprocessing.get_glove_embedding_BiLSTM(dictionary)
    

# %%
embedding_layer = Embedding(input_dim=len(dictionary), output_dim=100, input_length=150, weights = [embedding], trainable=False)
# %%
