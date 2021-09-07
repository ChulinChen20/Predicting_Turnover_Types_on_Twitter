import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pylab import barh,plot,yticks,show,xlabel,figure

# Get connections to the databases
path1 = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//LIWC2015 Results (Clean 2Class Subjects(-stay)).db'
db = sqlite3.connect(path1)

# Get the contents of a table
data = pd.read_sql("SELECT label, posemo, negemo, anx, anger, sad FROM turnover_tweets", db)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
print(data.head())
print(len(data))
cols = data.columns.tolist()
print(cols)


x_train = data.iloc[0:12826, 1:6]  # original: [0:18089, 1:9]
print(x_train.head())
y_train = data.iloc[0:12826, 0]  # [0:18089, 0]
x_test = data.iloc[12826:, 1:6]  # [18089:25842, 1:9]
y_test = data.iloc[12826:, 0]  # [18089:25842, 0]

# Standaradization: to avoid bias toward a certain feature
sc = StandardScaler(with_mean=False)
sc.fit(x_train)

x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)

# compute chi2 for each feature (x = document term matrix)
chi2score, pval = chi2(x_train_nor, y_train)
chi2values = ["{0:.7f}".format(x)for x in chi2score]
pvalues=["{0:.7f}".format(x)for x in pval]
print(chi2values)
print(pvalues)

# Two features with highest chi-squared statistics are selected
chi2_features = SelectKBest(chi2, k=5)
X_kbest_features = chi2_features.fit_transform(x_train_nor, y_train)


# Reduced features
print('Original feature number:', x_train_nor.shape[1])
print('Reduced feature number:', X_kbest_features.shape[1])

import scipy.stats as stats
print('means for posemo:', data.groupby(['label'])['posemo'].mean())
print('means for negemo:', data.groupby(['label'])['negemo'].mean())
print('means for anx:', data.groupby(['label'])['anx'].mean())
print('means for anger:', data.groupby(['label'])['anger'].mean())
print('means for sad:', data.groupby(['label'])['sad'].mean())

# plot top features
figure(figsize=(6,6))
wscores = zip(['posemo', 'negemo', 'anx', 'anger', 'sad'],chi2score)
wchi2 = sorted(wscores,key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-25:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
barh(x,topchi2[1],align='center',alpha=.2,color='g')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
yticks(x,labels)
xlabel('$\chi^2$')
show()




