import pandas as pd
import sqlite3
import re
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures
from nltk.collocations import QuadgramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures
import numpy as np

# connect to db
path = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//Training data.db'
conn = sqlite3.connect(path, isolation_level = None)

# read into pandas
df = pd.read_sql("SELECT username, tweetText FROM turnover_tweets", conn)
pd.set_option('display.max_colwidth', -1)

print(df.tweetText.head())

df = df.iloc[0:12827, :]

tweet_list = df['tweetText'].tolist()

# find bigrams
finder = BigramCollocationFinder.from_documents(item.split() for item in tweet_list)
bigram_measures = BigramAssocMeasures()
finder.apply_freq_filter(128) # exclude frequency of less than 1% of user usage
finder.score_ngrams(bigram_measures.pmi)
yield_bigrams = finder.above_score(bigram_measures.pmi, 4)  # get generator object for bigram with pmi > = 4

n = 0
bigram_list = []
for el in yield_bigrams:  # return values in generator object
    bigram_list.append(el)
    n += 1

print("number of bigrams:" + str(n))
print(finder.nbest(bigram_measures.pmi, 2000))  # top 100 bigrams by pmi

with open('Bigrams(128)', "w") as output:
    output.write(str(bigram_list))

data = np.array(bigram_list)
np.savez("Bigrams(128)", data)


# find trigrams
finder2 = TrigramCollocationFinder.from_documents(item.split() for item in tweet_list)
trigram_measures = TrigramAssocMeasures()
finder2.apply_freq_filter(128) # exclude frequency of less than 1% of user usage
finder2.score_ngrams(trigram_measures.pmi)
yield_trigrams = finder2.above_score(trigram_measures.pmi, 6)

m = 0
trigram_list = []
for el in yield_trigrams:
    trigram_list.append(el)
    m += 1

print("number of trigrams:" + str(m))
print(finder2.nbest(trigram_measures.pmi, 2000))  # top 100 bigrams by pmi

with open('Trigrams(128)', "w") as output:
    output.write(str(trigram_list))

data = np.array(trigram_list)
np.savez("Trigrams(128)", data)

# find quadgrams
finder3 = QuadgramCollocationFinder.from_documents(item.split() for item in tweet_list)
quadgram_measures = QuadgramAssocMeasures()
finder3.apply_freq_filter(128) # exclude frequency of less than 1% of user usage
finder3.score_ngrams(quadgram_measures.pmi)
yield_quadgrams = finder2.above_score(quadgram_measures.pmi, 8)

m = 0
quadgram_list = []
for el in yield_quadgrams:
    quadgram_list.append(el)
    m += 1

print("number of quadgrams:" + str(m))
print(finder3.nbest(quadgram_measures.pmi, 2000))  # top 100 bigrams by pmi

with open('quadgrams(128)', "w") as output:
    output.write(str(quadgram_list))

data = np.array(quadgram_list)
np.savez("Quadgrams(128)", data)