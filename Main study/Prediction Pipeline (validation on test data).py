import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sqlite3
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


# Get connections to the databases
path1 = 'LIWC2015 Results.db'  
db = sqlite3.connect(path1)

# Get the contents of a table
data = pd.read_sql('SELECT label, posemo, negemo, anx, anger, sad FROM turnover_tweets',  db)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
print(data.tail())
print(len(data))
cols = data.columns.tolist()
print(cols)


# encode labels
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
ct = ColumnTransformer([("label", OneHotEncoder(),[0])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
print(data.label.head())
ct.fit_transform(data)

# get ngrams
ngrams = joblib.load('4grams Kbest Matrix(50000) Dataframe.pkl')
print(ngrams.tail())
print(len(ngrams))

# Load LDA MAtrix
topics = joblib.load('/Users/chulinchen/PycharmProjects/Turnover Project/Analysis/'
                     'Kbest Topics Matrix(58-4)_Gensim(n=62).pkl')
print(topics)
print(len(topics))

df = pd.concat([data, ngrams, topics], axis=1)


all_features = df.iloc[:, 1:]
dv = df.iloc[:, 0]
feature_names = all_features.columns.tolist()
print(df.tail())
print(len(df))



def get_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    r2 = r2_score(true_labels, predicted_labels, multioutput='raw_values')
    return accuracy, precision, recall, f1


def cross_validation_scores_list(metrics_list, fold):
    accuracy, precision, recall, f1 = [], [], [], []
    for (a, b, c, d) in metrics_list:
        accuracy.append(a)
        precision.append(b)
        recall.append(c)
        f1.append(d)

    return [accuracy, precision, recall, f1]


def show_confusion_matrix(y_true, y_test, label):
    cm = confusion_matrix(y_true, y_test)
    pd.DataFrame(cm).to_csv("confusion_matrix_" + str(label) + ".csv")
    print("%s's confusion matrix:" % label)
    print(cm)


# testing models:
accuracy, precision, f1, recall = 0, 0, 0, 0
logr_metrics, mnb_metrics, xgb_metrics, svm_metrics = [], [], [], []
logr_r2, mnb_r2, xgb_r2, svm_r2 = [], [], [], []

# store all results for confusion matrix:
all_y_test = []
all_lr = []
all_mnb = []
all_xgb = []
all_svm = []

x_train = df.iloc[0:12826, 1:]
y_train = data.iloc[0:12826, 0]
x_test = df.iloc[12826:, 1:]
y_test = data.iloc[12826:, 0]

# Standaradization: to avoid bias toward a certain feature
sc = StandardScaler(with_mean=False)
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


all_y_test.extend(y_test)

# store all results for confusion matrix:
logr = LogisticRegression(random_state=42, max_iter=100000)
logr.fit(x_train, y_train)
y_pred1 = logr.predict(x_test)
metrics1 = get_metrics(y_test, y_pred1)
logr_metrics.append(metrics1)
R2_lr = r2_score(y_test, y_pred1)
logr_r2.append(R2_lr)
yPred_p1 = logr.predict_proba(x_test)[:,1]
auc_lr = roc_auc_score(y_test, yPred_p1)
all_lr.extend(y_pred1)
print(metrics1)
print("Logistic regression completed")


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
metrics2 = get_metrics(y_test, y_pred2)
mnb_metrics.append(metrics2)
R2_mnb = r2_score(y_test, y_pred2)
mnb_r2.append(R2_mnb)
yPred_p2 = mnb.predict_proba(x_test)[:,1]
auc_mnb = roc_auc_score(y_test, yPred_p2)
all_mnb.extend(y_pred2)
print(metrics2)
print("Naive Bayes completed")


xgb = XGBClassifier(random_state=42)
xgb.fit(x_train, y_train)
y_pred3 = xgb.predict(x_test)
metrics3 = get_metrics(y_test, y_pred3)
xgb_metrics.append(metrics3)
R2_xgb = r2_score(y_test, y_pred3)
xgb_r2.append(R2_xgb)
yPred_p3 = xgb.predict_proba(x_test)[:,1]
auc_xgb = roc_auc_score(y_test, yPred_p3)
all_xgb.extend(y_pred3)
print(metrics3)
print("XGBoost completed")


svm = SVC(max_iter=100000, probability=True)
svm.fit(x_train, y_train)
y_pred4 = svm.predict(x_test)
metrics4 = get_metrics(y_test, y_pred4)
svm_metrics.append(metrics4)
R2_svm = r2_score(y_test, y_pred4)
svm_r2.append(R2_svm)
yPred_p4 = svm.predict_proba(x_test)[:,1]
auc_svm = roc_auc_score(y_test, yPred_p4)
all_svm.extend(y_pred4)
print(metrics4)
print("SVM completed")


# get validation on test data results:
crossed_logr_metrics = cross_validation_scores_list(logr_metrics, 5)
crossed_mnb_metrics = cross_validation_scores_list(mnb_metrics, 5)
crossed_xgb_metrics = cross_validation_scores_list(xgb_metrics, 5)
crossed_svm_metrics = cross_validation_scores_list(svm_metrics, 5)


# print the results:
print("Results for 5-fold cross-validation:")

ls1 = ["{:.4f}".format(np.mean(crossed_logr_metrics[0])), "{:.4f}".format(np.mean(crossed_logr_metrics[1])),
       "{:.4f}".format(np.mean(crossed_logr_metrics[2])), "{:.4f}".format(np.mean(crossed_logr_metrics[3]))
       ]

ls2 = ["{:.4f}".format(np.mean(crossed_mnb_metrics[0])), "{:.4f}".format(np.mean(crossed_mnb_metrics[1])),
       "{:.4f}".format(np.mean(crossed_mnb_metrics[2])), "{:.4f}".format(np.mean(crossed_mnb_metrics[3]))
      ]

ls3 = ["{:.4f}".format(np.mean(crossed_xgb_metrics[0])), "{:.4f}".format(np.mean(crossed_xgb_metrics[1])),
       "{:.4f}".format(np.mean(crossed_xgb_metrics[2])), "{:.4f}".format(np.mean(crossed_xgb_metrics[3]))
      ]


ls4 = ["{:.4f}".format(np.mean(crossed_svm_metrics[0])), "{:.4f}".format(np.mean(crossed_svm_metrics[1])),
       "{:.4f}".format(np.mean(crossed_svm_metrics[2])), "{:.4f}".format(np.mean(crossed_svm_metrics[3]))
       ]


cross_val_df = pd.DataFrame(
    {"Logistic Regression": ls1,
     "Multinomial Naïve Bayes": ls2,
     "XGBoost Decision Tree": ls3,
     "SVM": ls4,
     })
cross_val_df.index = ["Accuracy", "Precision", "Recall", "F1"]
print(cross_val_df)
joblib.dump(cross_val_df, "cross-validated matrics (Validation)")

# print the confusion matrix:
show_confusion_matrix(all_y_test, all_lr, "LR")
show_confusion_matrix(all_y_test, all_mnb, "mnb")
show_confusion_matrix(all_y_test, all_xgb, "xgb")
show_confusion_matrix(all_y_test, all_svm, "svm")

# print R2
print("Results for R-squared on test data:")
R2_table_test = list(zip(logr_r2, mnb_r2, xgb_r2, svm_r2))
table_test = pd.DataFrame(R2_table_test)
table_test.columns = ['Logistic Regression','Multinomial Naïve Bayes', 'XGBoost Decision Tree',"SVM"]
print(table_test)
joblib.dump(table_test, "R2 for test data")

# print AUC
print("Results for AUC:")
table_AUC = pd.DataFrame(
    {"Logistic Regression": auc_lr,
     "Multinomial Naïve Bayes": auc_mnb,
     "XGBoost Decision Tree": auc_xgb,
     "SVM": auc_svm,
     }, columns = ['Logistic Regression','Multinomial Naïve Bayes', 'XGBoost Decision Tree',"SVM"], index=[0])

print(table_AUC)
joblib.dump(table_AUC, "AUC (Validation)")


