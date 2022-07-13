import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import metrics as pf
from utils import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#####SVM WITHOUT CONTEXT
x_train_df = pd.read_csv('data/x_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
x_test_df = pd.read_csv('data/x_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_train_df = pd.read_csv('data/y_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_test_df = pd.read_csv('data/y_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})

x_train = x_train_df['sar_text']
x_test = x_test_df['sar_text']
y_train = y_train_df['label']
y_test = y_test_df['label']

tfidf = TfidfVectorizer()
tfidf_x_train = tfidf.fit_transform(x_train)
tfidf_x_test = tfidf.transform(x_test)

svm_classifier = svm.LinearSVC().fit(tfidf_x_train, y_train)
tfidf_y_pred = svm_classifier.predict(tfidf_x_test)

pf.plot_coefficients(svm_classifier, tfidf.get_feature_names_out())

metrics_tfidf = pf.metrics(y_test, tfidf_y_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_tfidf)

df_tfidf = pd.DataFrame({'sar_id': x_test_df['sar_id'], 'label': y_test, 'prediction': tfidf_y_pred})
pf.json_metrics("data\SVMmetrics_without_context.json", "SVM - LinearSVC without context", "TF-IDF", metrics_tfidf, df_tfidf)


####SVM WITH CONTEXT
x_train_df = pd.read_csv('data/x_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
x_test_df = pd.read_csv('data/x_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_train_df = pd.read_csv('data/y_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_test_df = pd.read_csv('data/y_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})

x_train = x_train_df.loc[:, ['sar_text', 'obl_text', 'eli_text']]
x_test = x_test_df.loc[:, ['sar_text', 'obl_text', 'eli_text']]
y_train = y_train_df['label']
y_test = y_test_df['label']

tf_idf_train, tf_idf_test = preprocessing.get_tfidf_context(x_train, x_test)

svm_classifier = svm.LinearSVC().fit(tf_idf_train, y_train)
tfidf_pred = svm_classifier.predict(tf_idf_test)

#pf.plot_coefficients(svm_classifier, tfidf.get_feature_names_out())

metrics_context = pf.metrics(y_test, tfidf_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_context)

df = pd.DataFrame({'sar_id': x_test_df['sar_id'], 'label': y_test, 'predicted_value': tfidf_pred})
pf.json_metrics("data\SVMmetrics_context.json", "SVM - LinearSVC with context", "TF-IDF", metrics_context, df)


#### GLOVE EMBEDDING WITHOUT CONTEXT
x_train_df = pd.read_csv('data/x_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
x_test_df = pd.read_csv('data/x_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_train_df = pd.read_csv('data/y_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_test_df = pd.read_csv('data/y_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})

x_train = x_train_df.loc[:, ['sar_text']]
x_test = x_test_df.loc[:, ['sar_text']]
y_train = y_train_df['label']
y_test = y_test_df['label']

glove_train, glove_test = preprocessing.get_glove_embedding_SVM(x_train, x_test)

#svm_classifier = make_pipeline(StandardScaler(), svm.LinearSVC(max_iter=10000))
#svm_classifier.fit(glove_train, y_train)

svm_classifier = svm.LinearSVC(dual=False).fit(glove_train, y_train)
glove_test_pred = svm_classifier.predict(glove_test)

metrics_context = pf.metrics(y_test, glove_test_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_context)

df = pd.DataFrame({'sar_id': x_test_df['sar_id'], 'label': y_test, 'predicted_value': glove_test_pred})
pf.json_metrics("data\SVMmetrics_GloVe.json", "SVM - LinearSVC without context", "Glove", metrics_context, df)


#### GLOVE EMBEDDING WITH CONTEXT
x_train_df = pd.read_csv('data/x_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
x_test_df = pd.read_csv('data/x_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_train_df = pd.read_csv('data/y_train.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})
y_test_df = pd.read_csv('data/y_test.csv', converters={'sar_text' : str, 'obl_text' : str, 'eli_text' : str})

x_train = x_train_df.loc[:, ['sar_text', 'obl_text', 'eli_text']]
x_test = x_test_df.loc[:, ['sar_text', 'obl_text', 'eli_text']]
y_train = y_train_df['label']
y_test = y_test_df['label']

glove_train, glove_test = preprocessing.get_glove_embedding_SVM(x_train.loc[:, ['sar_text', 'obl_text', 'eli_text']], x_test.loc[:, ['sar_text', 'obl_text', 'eli_text']])

svm_classifier = svm.LinearSVC(dual=False).fit(glove_train, y_train)
glove_test_pred = svm_classifier.predict(glove_test)

metrics_context = pf.metrics(y_test, glove_test_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_context)

df = pd.DataFrame({'sar_id': x_test_df['sar_id'], 'label': y_test, 'predicted_value': glove_test_pred})
pf.json_metrics("data\SVMmetrics_GloVe_context.json", "SVM - LinearSVC with context", "Glove", metrics_context, df)