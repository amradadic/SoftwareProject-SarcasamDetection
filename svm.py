import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import predictions as pf
import numpy as np
from utils import preprocessing
from sklearn.model_selection import train_test_split
import math
from sklearn import metrics as m
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df_SPIRS_sarcastic = pd.read_csv('data/SPIRS-sarcastic2.csv')
df_SPIRS_non_sarcastic = pd.read_csv('data/SPIRS-non-sarcastic2.csv')

'''
#####SVM WITHOUT CONTEXT
df_SPIRS = pf.make_dataframe(df_SPIRS_sarcastic, df_SPIRS_non_sarcastic)

df_SPIRS = df_SPIRS.sample(frac=1) #shuffle
x_train, x_test, y_train, y_test = train_test_split(df_SPIRS['sar_text'].values, df_SPIRS['label'].values, test_size=0.2, shuffle=False)

tfidf = TfidfVectorizer()
tfidf_x_train = tfidf.fit_transform(x_train)
tfidf_x_test = tfidf.transform(x_test)

svm_classifier = svm.LinearSVC().fit(tfidf_x_train, y_train)
tfidf_y_pred = svm_classifier.predict(tfidf_x_test)

pf.plot_coefficients(svm_classifier, tfidf.get_feature_names_out())

metrics_tfidf = pf.metrics(y_test, tfidf_y_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_tfidf)

n_train = math.floor(0.8 * df_SPIRS['label'].shape[0])
df_tfidf = pd.DataFrame({'sar_id': df_SPIRS['sar_id'][n_train:].values, 'label': y_test, 'prediction': tfidf_y_pred})
pf.json_metrics("json\SVMmetrics2.json", "SVM - LinearSVC only sar_text", "TF-IDF", metrics_tfidf, df_tfidf)


####SVM WITH CONTEXT
df_SPIRS = pf.make_dataframe(df_SPIRS_sarcastic, df_SPIRS_non_sarcastic, context=True)

df_SPIRS = df_SPIRS.sample(frac=1) #shuffle
x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, ['sar_text', 'obl_text', 'eli_text']], df_SPIRS['label'], test_size=0.2, shuffle=False)

tf_idf_train, tf_idf_test = preprocessing.get_tfidf_context(x_train, x_test)

svm_classifier = svm.LinearSVC().fit(tf_idf_train, y_train)
tfidf_pred = svm_classifier.predict(tf_idf_test)

#pf.plot_coefficients(svm_classifier, tfidf.get_feature_names_out())

metrics_context = pf.metrics(y_test, tfidf_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_context)

n_train = math.floor(0.8 * df_SPIRS['label'].shape[0])

df = pd.DataFrame({'sar_id': df_SPIRS['sar_id'][n_train:], 'label': y_test, 'predicted_value': tfidf_pred})
pf.json_metrics("json\SVMmetrics2_context.json", "SVM - LinearSVC with context", "TF-IDF", metrics_context, df)


#### GLOVE EMBEDDING WITHOUT CONTEXT
df_SPIRS = pf.make_dataframe(df_SPIRS_sarcastic, df_SPIRS_non_sarcastic)

df_SPIRS = df_SPIRS.sample(frac=1) #shuffle
x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, ['sar_text']], df_SPIRS['label'], test_size=0.2, shuffle=False)

glove_train, glove_test = preprocessing.get_glove_embedding_SVM(x_train, x_test)

#svm_classifier = make_pipeline(StandardScaler(), svm.LinearSVC(max_iter=10000))
#svm_classifier.fit(glove_train, y_train)

svm_classifier = svm.LinearSVC(dual=False).fit(glove_train, y_train)
glove_test_pred = svm_classifier.predict(glove_test)

#pf.plot_coefficients(svm_classifier, )

metrics_context = pf.metrics(y_test, glove_test_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_context)

n_train = math.floor(0.8 * df_SPIRS['label'].shape[0])
df = pd.DataFrame({'sar_id': df_SPIRS['sar_id'][n_train:].values, 'label': y_test, 'predicted_value': glove_test_pred})
pf.json_metrics("json\SVMmetrics_GloVe.json", "SVM - LinearSVC without context", "Glove", metrics_context, df)
'''

#### GLOVE EMBEDDING WITH CONTEXT
df_SPIRS = pf.make_dataframe(df_SPIRS_sarcastic, df_SPIRS_non_sarcastic, context=True)

df_SPIRS = df_SPIRS.sample(frac=1) #shuffle
x_train, x_test, y_train, y_test = train_test_split(df_SPIRS.loc[:, ['sar_text', 'obl_text', 'eli_text']], df_SPIRS['label'], test_size=0.2, shuffle=False)

glove_train, glove_test = preprocessing.get_glove_embedding_SVM(x_train.loc[:, ['sar_text', 'obl_text', 'eli_text']], x_test.loc[:, ['sar_text', 'obl_text', 'eli_text']])

svm_classifier = svm.LinearSVC(dual=False).fit(glove_train, y_train)
glove_test_pred = svm_classifier.predict(glove_test)

#pf.plot_coefficients(svm_classifier, )

metrics_context = pf.metrics(y_test, glove_test_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_context)

n_train = math.floor(0.8 * df_SPIRS['label'].shape[0])
df = pd.DataFrame({'sar_id': df_SPIRS['sar_id'][n_train:], 'label': y_test, 'predicted_value': glove_test_pred})
pf.json_metrics("json\SVMmetrics_GloVe_context.json", "SVM - LinearSVC with context", "Glove", metrics_context, df)
