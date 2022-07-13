import numpy as np
import json
from os import path
from matplotlib import pyplot as plt
from sklearn import metrics as m

def json_metrics(file_name, prediction_model, embedding, metrics, df):
    dictionary = {'Model': prediction_model,
                  'User embedding': embedding,
                  'Metrics': metrics,
                  'Data': df.to_dict('records')}

    if path.isfile(file_name):  # file exist
        with open(file_name) as fp:
            listObj = json.load(fp)

        listObj.append(dictionary)

        with open(file_name, 'w') as json_file:
            json.dump(listObj, json_file, indent=4)
    else:
        with open(file_name, 'w') as json_file:
            json.dump([dictionary], json_file, indent=4)


def metrics(y_test, y_pred, target_names):
    tn, fp, fn, tp = m.confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    dict_confusion = {'True negative': int(tn),
                      'False positive': int(fp),
                      'False negative': int(fn),
                      'True positive': int(tp),
                      }
    dict_report = m.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return {**dict_confusion, **dict_report}


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
