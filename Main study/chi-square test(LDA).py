import pandas as pd
import pandas as np
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pylab import barh,plot,yticks,show,grid,xlabel,figure
from sklearn.externals import joblib

# Get connections to the databases
path1 = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Quitters//Clean 2Class Subjects(-stay).db'
db = sqlite3.connect(path1)

# Get the contents of a table
data = pd.read_sql('SELECT tweetText, label FROM turnover_tweets',  db)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10000)

# load LDA matrix
LDA_matrix_df = joblib.load('Gensim LDA matrix_final(n=62)_whole.pkl')
print(LDA_matrix_df)
topics = LDA_matrix_df.columns.tolist()



x_train = LDA_matrix_df[0:12826]
print(x_train.head())
y_train = data.iloc[0:12826, 1]
x_test = LDA_matrix_df[12826:]
y_test = data.iloc[12826:,1]

# mean percentage by turnover types for each topic
whole = pd.concat([x_train,  y_train], axis=1)
print(whole)
plan_mean = whole.loc[whole['label'] == "1"].mean().reset_index(level=-1)
plan_mean.columns = ['topic', 'plan_mean_percentage']
diss_mean = whole.loc[whole['label'] == "3"].mean().reset_index(level=-1)
diss_mean.columns = ['topic', 'diss_mean_percentage']

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
table = list(zip(topics, chi2score,pvalues))
table = pd.DataFrame(table)
table.columns = ['topic','chi2score', 'pvalues']
table.to_csv('chi_table_topics.csv', index=False)
sig_table = table.loc[table['pvalues'].astype('float') <= 0.05]
sig_table = pd.merge(sig_table, plan_mean, on="topic", how="left")
sig_table = pd.merge(sig_table, diss_mean, on="topic", how="left")
print(len(sig_table))
sig_table.to_csv('chi_results_topics.csv', index=False)

# Two features with highest chi-squared statistics are selected
selector = SelectKBest(chi2, k=58)
kbest_features_train = selector.fit_transform(x_train_nor, y_train)


# get feature names & dataframe for kbest
mask = selector.get_support() #list of booleans
new_features = []  # The list of your K best features

for bool, feature in zip(mask, topics):
    if bool:
        new_features.append(feature)

print(new_features)
kbest_train_dataframe = pd.DataFrame(kbest_features_train, columns=new_features)

# sort features with descending importance
names = x_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True]).reset_index(drop=True)
print(ns_df_sorted)


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

print(len(kbest_matrix))

# remove uninterpretable and not coherent topics
del kbest_matrix['Topic22']
del kbest_matrix['Topic25']
del kbest_matrix['Topic47']
del kbest_matrix['Topic55']
print(kbest_matrix.head())
print(kbest_matrix.tail())

#  automatically split the model file into pickled numpy array files if model size is large
joblib.dump(kbest_matrix, '/Users/chulinchen/PycharmProjects/Turnover Project/Analysis/'
                          'Kbest Topics Matrix(58-4)_Gensim(n=62).pkl')


# plot top features
figure(figsize=(6,6))
wscores = zip(topics,chi2score)
wchi2 = sorted(wscores,key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-25:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
barh(x,topchi2[1],align='center',alpha=.2,color='g')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
yticks(x,labels)
xlabel('$\chi^2$')
show()
