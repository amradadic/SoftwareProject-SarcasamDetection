#%%
import pandas as pd
from utils import preprocessing
from sklearn.model_selection import train_test_split
from utils import predictions as pf
from sklearn import svm
from sklearn import metrics as m


df_SPIRS_sarcastic = pd.read_csv('SPIRS-sarcastic.csv')
df_SPIRS_non_sarcastic = pd.read_csv('SPIRS-non-sarcastic.csv')

df_SPIRS_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_sarcastic, 'sar_text')
df_SPIRS_non_sarcastic = preprocessing.remove_na_from_column(df_SPIRS_non_sarcastic, 'sar_text')


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

df_SPIRS_sarcastic = df_SPIRS_sarcastic[['sar_text']]
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic[['sar_text']]

df_SPIRS_sarcastic = df_SPIRS_sarcastic.assign(label=1)
df_SPIRS_non_sarcastic = df_SPIRS_non_sarcastic.assign(label=0)

df_SPIRS = pd.concat([df_SPIRS_sarcastic, df_SPIRS_non_sarcastic], ignore_index=True)

x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, df_SPIRS.columns != 'label'], df_SPIRS[['label']], test_size=0.2, random_state=123, shuffle=True)

glove_train, glove_test = preprocessing.get_glove_embedding(x_train['sar_text'], x_test['sar_text'])
print(glove_train.shape,  glove_test.shape)

svm_classifier = svm.LinearSVC().fit(glove_train, y_train['label'])
glove_test_pred = svm_classifier.predict(glove_test)

print(m.classification_report(y_test, glove_test_pred, target_names=['Non-sarcastic', 'Sarcastic']))

# %%
