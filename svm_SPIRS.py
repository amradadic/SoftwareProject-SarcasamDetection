import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import predictions as pf

df_SPIRS_sarcastic = pd.read_csv('SPIRS-sarcastic.csv')
df_SPIRS_non_sarcastic = pd.read_csv('SPIRS-non-sarcastic.csv')

df_SPIRS = pf.preparing_data(df_SPIRS_sarcastic, df_SPIRS_non_sarcastic)
x_train, x_test, y_train, y_test, tweet_id = pf.train_test_split(df_SPIRS)

#######CountVectorizer
cv = CountVectorizer()
cv_x_train = cv.fit_transform(x_train)
cv_x_test = cv.transform(x_test)

svm_classifier = svm.SVC(kernel='linear').fit(cv_x_train, y_train)
y_pred = svm_classifier.predict(cv_x_test)

metrics_cv = pf.metrics(y_test, y_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_cv)

df_cv = pd.DataFrame({'tweet_id': tweet_id, 'label': y_test, 'prediction': y_pred})
pf.json_predictions("predictions.json", "SVM", "CountVectorizer", metrics_cv, df_cv)


#######TFIDF
tfidf = TfidfVectorizer(ngram_range=(1, 3),)
tfidf_x_train = tfidf.fit_transform(x_train)
tfidf_x_test = tfidf.transform(x_test)

svm_classifier = svm.SVC(kernel='linear').fit(tfidf_x_train, y_train)

tfidf_y_pred = svm_classifier.predict(tfidf_x_test)

metrics_tfidf = pf.metrics(y_test, tfidf_y_pred, target_names=['Non-sarcastic', 'Sarcastic'])
print(metrics_tfidf)

df_tfidf = pd.DataFrame({'tweet_id': tweet_id,'label': y_test, 'prediction': tfidf_y_pred})
pf.json_predictions("predictions.json", "SVM", "TF-IDF", metrics_tfidf, df_tfidf)