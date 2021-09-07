import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler
import sqlite3
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

'''
# Feature extraction
# Feature extractor is always based on the training data, and will never change on newer documents (p.181)
# N-grams
# Get connections to the databases
path1 = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//LIWC2015 Results (Clean 2Class Subjects(-stay)).db'  # LIWC(Clean 2class Subjects).db'
db = sqlite3.connect(path1)

# Get the contents of a table
data = pd.read_sql('SELECT label, posemo, negemo, anx, anger, sad FROM turnover_tweets',  db)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
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
                     'Kbest Topics Matrix(58-4)_Gensim(n=62).pkl')  #'Kbest Topics Matrix(80)_Gensim(Uni, n=100).pkl'
print(topics)
print(len(topics))

df = pd.concat([data, ngrams, topics], axis=1)
df = df.iloc[0:12826, :]
'''


df = joblib.load( 'final training data (50000).pkl')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10000)
all_features = df.iloc[:, 1:]
dv = df.iloc[:, 0]
feature_names = all_features.columns.tolist()
print(df.tail())
print(len(df))

# Stack all three arrays

def get_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    r2 = r2_score(true_labels, predicted_labels, multioutput='raw_values')
    return accuracy, precision, recall, f1

'''
def get_R2(model, true_labels, predicted_labels):
    R2_mean = statistics.mean(model.score(true_labels, predicted_labels))
    R2_SD = statistics.stdev(model.score(true_labels, predicted_labels))
    R2_min = min((model.score(true_labels, predicted_labels)))
    R2_max = max((model.score(true_labels, predicted_labels)))
    return R2_mean, R2_SD, R2_min, R2_max
'''

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
logr_r2_adjusted, mnb_r2_adjusted, xgb_r2_adjusted, svm_r2_adjusted = [], [], [], []

# store all results for confusion matrix:
all_y_test = []
all_lr = []
all_mnb = []
all_xgb = []
all_svm = []

# Standaradization: to avoid bias toward a certain feature
sc = StandardScaler(with_mean=False)

n = 5
cv = KFold(n_splits=n, random_state=42, shuffle=True)
round = 0

for train_index, test_index in cv.split(all_features, dv):
    x_train, x_test = all_features.iloc[train_index], all_features.iloc[test_index]
    y_train, y_test = dv.iloc[train_index], dv.iloc[test_index]
    scaler = sc.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    all_y_test.extend(y_test)

# store all results for confusion matrix:

# cross-validation:

    logr = LogisticRegression(random_state=42, max_iter=100000)
    logr.fit(x_train, y_train)
    y_pred1 = logr.predict(x_test)
    metrics1 = get_metrics(y_test, y_pred1)
    logr_metrics.append(metrics1)
    # R2_lr = r2_score(y_test, y_pred1)
    # R2_lr_train = r2_score(y_train, logr.predict(x_train))
    R2_lr = logr.score(x_train, y_train)
    logr_r2.append(R2_lr)

    R2_a_lr = 1 - (1 - logr.score(x_train, y_train) * (len(y_train) - 1)
                   / (len(y_train.values.ravel()) - x_train.shape[1] - 1))
    logr_r2_adjusted.append(R2_a_lr)
    all_lr.extend(y_pred1)
    print("metrics:",metrics1)
    print("R2:", R2_lr)
    print("Adjusted R2:", R2_a_lr)
    print("Logistic regression completed")

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_pred2 = mnb.predict(x_test)
    metrics2 = get_metrics(y_test, y_pred2)
    mnb_metrics.append(metrics2)
    # R2_mnb = r2_score(y_test, y_pred2)
    # R2_mnb_train = r2_score(y_train, mnb.predict(x_train))
    R2_mnb = mnb.score(x_train, y_train)
    mnb_r2.append(R2_mnb)
    R2_a_mnb = 1 - (1 - mnb.score(x_train, y_train) * (len(y_train) - 1)
                    / (len(y_train.values.ravel()) - x_train.shape[1] - 1))
    mnb_r2_adjusted.append(R2_a_mnb)
    all_mnb.extend(y_pred2)
    print("metrics:",metrics2)
    print("R2:", R2_mnb)
    print("Adjusted R2:", R2_a_mnb)
    print("Naive Bayes completed")

    xgb = XGBClassifier(random_state=42)
    xgb.fit(x_train, y_train)
    y_pred3 = xgb.predict(x_test)
    metrics3 = get_metrics(y_test, y_pred3)
    xgb_metrics.append(metrics3)
    R2_xgb = xgb.score(x_train, y_train)
    xgb_r2.append(R2_xgb)
    R2_a_xgb = 1 - (1 - xgb.score(x_train, y_train) * (len(y_train) - 1)
                    / (len(y_train.values.ravel()) - x_train.shape[1] - 1))
    xgb_r2_adjusted.append(R2_a_xgb)
    all_xgb.extend(y_pred3)
    print("metrics:",metrics3)
    print("R2:", R2_xgb)
    print("Adjusted R2:", R2_a_xgb)

    svm = SVC(max_iter=100000)
    svm.fit(x_train, y_train)
    y_pred4 = svm.predict(x_test)
    metrics4 = get_metrics(y_test, y_pred4)
    svm_metrics.append(metrics4)
    R2_svm = svm.score(x_train, y_train)
    svm_r2.append(R2_svm)
    R2_a_svm = 1 - (1 - svm.score(x_train, y_train) * (len(y_train) - 1)
                    / (len(y_train.values.ravel()) - x_train.shape[1] - 1))
    svm_r2_adjusted.append(R2_a_svm)
    all_svm.extend(y_pred4)
    print("metrics:",metrics4)
    print("R2:", R2_svm)
    print("Adjusted R2:", R2_a_svm)
    print("SVM completed")
    round = round + 1
    print(round, "round has finished")

# get cross validation results:
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
joblib.dump(cross_val_df, "cross-validated matrics")

# print the confusion matrix:
show_confusion_matrix(all_y_test, all_lr, "LR")
show_confusion_matrix(all_y_test, all_mnb, "mnb")
show_confusion_matrix(all_y_test, all_xgb, "xgb")
show_confusion_matrix(all_y_test, all_svm, "svm")

# print R2
print("Results for 5-fold R-squared on test data:")
R2_table_test = list(zip(logr_r2, mnb_r2, xgb_r2, svm_r2))
table_test = pd.DataFrame(R2_table_test)
table_test.columns = ['Logistic Regression','Multinomial Naïve Bayes', 'XGBoost Decision Tree',"SVM"]
table_test.loc['mean'] = table_test.iloc[0:4].mean()
table_test.loc['SD'] = table_test.iloc[0:4].std()
table_test.loc['min'] = table_test.iloc[0:4].min()
table_test.loc['max'] = table_test.iloc[0:4].max()
print(table_test)
joblib.dump(table_test, "R2 for test data")

print("Results for 5-fold R-squared on training data:")
R2_table_train = list(zip(logr_r2_adjusted, mnb_r2_adjusted, xgb_r2_adjusted, svm_r2_adjusted))
table_train = pd.DataFrame(R2_table_train)
table_train.columns = ['Logistic Regression','Multinomial Naïve Bayes', 'XGBoost Decision Tree',"SVM"]
table_train.loc['mean'] = table_train.iloc[0:4].mean()
table_train.loc['SD'] = table_train.iloc[0:4].std()
table_train.loc['min'] = table_train.iloc[0:4].min()
table_train.loc['max'] = table_train.iloc[0:4].max()
print(table_train)
joblib.dump(table_train, "R2 for training data")
'''
# print validation results on testing set
x_train_v = df.iloc[0:12826, 1:]
y_train_v = data.iloc[0:12826, 0]
x_test_v = df.iloc[12826:, 1:]
y_test_v = data.iloc[12826:, 0]

# Standaradization: to avoid bias toward a certain feature
sc = StandardScaler(with_mean=False)
sc.fit(x_train_v)

x_train_nor = sc.transform(x_train_v)
x_test_nor = sc.transform(x_test_v)

logr = LogisticRegression(random_state=42, max_iter=100000)
logr.fit(x_train_nor, y_train_v)
lr_metrics = get_metrics(y_test_v, logr.predict(x_train_nor))

mnb = MultinomialNB()
mnb.fit(x_train_nor, y_train_v)
mnb_metrics = get_metrics(y_test_v, logr.predict(x_train_nor))
'''