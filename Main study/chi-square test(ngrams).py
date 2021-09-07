import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pylab import barh,plot,yticks,show,grid,xlabel,figure
from sklearn.externals import joblib
from sqlalchemy import create_engine


cross = joblib.load( 'cross-validated matrics')
R2_test = joblib.load('R2 for test data')
R2_train = joblib.load('R2 for training data')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10000)
print (cross)
print(R2_test)
print(R2_train)

# Get connections to the databases
path1 = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Quitters//Clean 2Class Subjects(-stay).db'
db = sqlite3.connect(path1)

# Get the contents of a table
data = pd.read_sql('SELECT tweetText, label FROM turnover_tweets',  db)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10000)


# load Unigrams
a = np.load('Unigram Names(2Class)(Tfidf)(df=0.01).npz')
Unigrams = a['arr_0']
print(Unigrams)
print(len(Unigrams))

# load bigrams
b = np.load("//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//Bigrams(128).npz")
Bigrams = b['arr_0']
Bigrams = Bigrams.astype(str)
Bigrams = [' '.join(x) for x in Bigrams]
Bigrams = np.array(Bigrams)
print(Bigrams)
print(len(Bigrams))

# load Trigrams
c = np.load("//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//Trigrams(128).npz")
Trigrams = c['arr_0']
Trigrams = Trigrams.astype(str)
Trigrams = [' '.join(x) for x in Trigrams]
Trigrams = np.array(Trigrams)
print(Trigrams)
print(len(Trigrams))

# load Quadgrams
c = np.load("//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//Quadgrams(128).npz")
Quadgrams = c['arr_0']
Quadgrams = Quadgrams.astype(str)
Quadgrams = [' '.join(x) for x in Quadgrams]
Quadgrams = np.array(Quadgrams)
print(Quadgrams)
print(len(Quadgrams))

ngrams = np.hstack((Unigrams, Bigrams, Trigrams, Quadgrams))  # combine arrays linearly
#np.random.choice(Unigrams, 5, replace=False)   #
print(ngrams)
print(len(ngrams))
#joblib.dump(pd.DataFrame(ngrams), 'all ngrams Dataframe.pkl')

vectorizer = TfidfVectorizer(ngram_range=(1,4))
vectorizer.fit_transform(ngrams)
Ngram_Names = vectorizer.get_feature_names()

x_train = vectorizer.transform(data.iloc[0:12826, 0])
x_train = x_train.todense()
x_train = x_train.astype(float)
y_train = data.iloc[0:12826, 1]
x_test = vectorizer.transform(data.iloc[12826:, 0])
x_test = x_test.todense()
x_test = x_test.astype(float)
y_test = data.iloc[12826:, 1]

# Standaradization: to avoid bias toward a certain feature
sc = StandardScaler(with_mean=False)
sc.fit(x_train)

x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)


# convert numpy to dataframe for selector.get_support
def display_features(features, feature_names):
    dataframe = pd.DataFrame(data=features, columns = feature_names)
    return dataframe


df_x_train = display_features(x_train, Ngram_Names)

'''
# compute chi2 and pvalue for each feature (x = document term matrix)
chi2score, pval = chi2(x_train_nor, y_train)
chi2values = ["{0:.7f}".format(x)for x in chi2score]
pvalues=["{0:.7f}".format(x)for x in pval]
print(chi2score)
print(pvalues)
allnames = df_x_train.columns.values
table = list(zip(allnames, chi2score,pvalues))
table = pd.DataFrame(table)
table.columns = ['ngram','chi2score', 'pvalues']
table.to_csv('chi_table_ngrams.csv', index=False)
sig_table = table.loc[table['pvalues'].astype('float') <= 0.05]
print(len(sig_table))
sig_table.to_csv('chi_results_ngrams.csv', index=False)
'''


# k best features with highest chi-squared statistics are selected
selector = SelectKBest(chi2, k=50000)
kbest_features_train = selector.fit_transform(x_train_nor, y_train)


# get feature names & dataframe for kbest
mask = selector.get_support() #list of booleans
new_features = []  # The list of your K best features

for bool, feature in zip(mask, Ngram_Names):
    if bool:
        new_features.append(feature)

print(new_features)
kbest_train_dataframe = pd.DataFrame(kbest_features_train, columns=new_features)

# sort features with descending importance
names = df_x_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
pvalues = selector.pvalues_[selector.get_support()]
names_scores = list(zip(names, scores, pvalues))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores', 'pvalues'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names','pvalues'], ascending = [False, True, True]).reset_index(drop=True)
print(ns_df_sorted)

# pandas to sqlite
#engine = create_engine('sqlite:////Users//chulinchen//Documents//turnover - twitter analysis//4g_chi_150000.db', echo=False)
#ns_df_sorted.to_sql('turnover_tweets', con=engine, if_exists='replace', index=False)
#engine.execute('SELECT * FROM turnover_tweets').fetchall()

sig_table_2 = ns_df_sorted.loc[ns_df_sorted['pvalues'] <= 0.05]
print(len(sig_table_2))


# fit selector to testing data
kbest_features_test = selector.transform(x_test_nor)
kbest_test_dataframe = pd.DataFrame(kbest_features_test, columns=new_features)
print(kbest_test_dataframe.head())
print(kbest_test_dataframe.tail())
print(len(kbest_test_dataframe))

# save feature names & dataframe
names = np.array(new_features)
# columns in diff orders, set sort=F to avoid error
kbest_matrix = kbest_train_dataframe.append(kbest_test_dataframe, ignore_index=True, sort=False)
print(kbest_matrix.head())
print(kbest_matrix.tail())
print(len(kbest_matrix))

#kbest_matrix.to_csv('4grams Kbest Matrix(150608) Dataframe.csv', index=False)

#  automatically split the model file into pickled numpy array files if model size is large
#engine = create_engine('sqlite:////Users//chulinchen//Documents//turnover - twitter analysis//4g_150608.db', echo=False)
#kbest_matrix.to_sql('ngrams', con=engine, if_exists='replace', index=False)
#engine.execute('SELECT * FROM ngrams').fetchall()
joblib.dump(kbest_matrix, '4grams Kbest Matrix(50000) Dataframe.pkl')


'''

# plot top features
figure(figsize=(12,12))
wscores = zip(Ngram_Names,chi2score)
wchi2 = sorted(wscores,key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-50:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
barh(x,topchi2[1],align='center',alpha=.2,color='g')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
yticks(x,labels)
xlabel('$\chi^2$')
show()
'''


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


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


# Load LDA MAtrix
topics = joblib.load('/Users/chulinchen/PycharmProjects/Turnover Project/Analysis/'
                     'Kbest Topics Matrix(58-4)_Gensim(n=62).pkl')  #'Kbest Topics Matrix(80)_Gensim(Uni, n=100).pkl'
print(topics)
print(len(topics))

df = pd.concat([data, kbest_matrix, topics], axis=1)
df = df.iloc[0:12826, :]
print(df.head())
print(len(df))

joblib.dump(df, 'final training data (50000).pkl')