import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pylab import barh,plot,yticks,show,xlabel,figure
from sklearn.externals import joblib
from sqlalchemy import create_engine


# Get connections to the databases
path1 = 'all_data.db'
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
b = np.load("Bigrams(128).npz")
Bigrams = b['arr_0']
Bigrams = Bigrams.astype(str)
Bigrams = [' '.join(x) for x in Bigrams]
Bigrams = np.array(Bigrams)
print(Bigrams)
print(len(Bigrams))

# load Trigrams
c = np.load("Trigrams(128).npz")
Trigrams = c['arr_0']
Trigrams = Trigrams.astype(str)
Trigrams = [' '.join(x) for x in Trigrams]
Trigrams = np.array(Trigrams)
print(Trigrams)
print(len(Trigrams))

# load Quadgrams
c = np.load("Quadgrams(128).npz")
Quadgrams = c['arr_0']
Quadgrams = Quadgrams.astype(str)
Quadgrams = [' '.join(x) for x in Quadgrams]
Quadgrams = np.array(Quadgrams)
print(Quadgrams)
print(len(Quadgrams))

ngrams = np.hstack((Unigrams, Bigrams, Trigrams, Quadgrams))  # combine arrays linearly
print(ngrams)
print(len(ngrams))


# apply tfidf vectorizer on the corpus, which used ngrams that were extracted as vocabulary
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

# create table of feature name, chi-squared values, and pvalues for kbest features
names = df_x_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
pvalues = selector.pvalues_[selector.get_support()]
names_scores = list(zip(names, scores, pvalues))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores', 'pvalues'])

# sort features with descending importance
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names','pvalues'], ascending = [False, True, True]).reset_index(drop=True)
print(ns_df_sorted)


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

#  save top n-grams
engine = create_engine('sqlite:////Users//path//4g_150608.db', echo=False)
kbest_matrix.to_sql('ngrams', con=engine, if_exists='replace', index=False)
engine.execute('SELECT * FROM ngrams').fetchall()
joblib.dump(kbest_matrix, '4grams Kbest Matrix(50000) Dataframe.pkl')


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
