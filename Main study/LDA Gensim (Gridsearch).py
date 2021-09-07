import numpy as np
import pandas as pd
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import tqdm
import sqlite3
from sqlalchemy import create_engine
import csv


# connect to db
path = 'all_data.db'
conn = sqlite3.connect(path, isolation_level = None)

# read LIMIT n rows starting from OFFSET m in the large file with specified chunksize
df = pd.read_sql("SELECT username, tweetText, label FROM turnover_tweets", conn)
pd.set_option('display.max_colwidth', None)

# load Unigrams
a = np.load('Unigram Names(2Class)(Tfidf)(df=0.01).npz')
Unigrams = a['arr_0']
print('Unigrams:\n', Unigrams)
print('Number of Unigrams:\n', len(Unigrams))
print(nltk.pos_tag(Unigrams))

# filter Unigrams (no stopwords, only nouns)

# import stopword list (and extend the list if necessary)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
new_stop_words = ['like', 'make', 'time',  'use',  'hey', 'okay',
                   'thing', 'people',  'day', 'know', 'till',
                    'lot',  'dont', 'come', 'let', 'say', 'want', 'think', 'tell', 'look', 'talk',
                    'cant', 'thats',  'cuz', 'coz',  'wont', 'youre', 'whats', 'shes', 'havent', 'theyre', 'wasnt',
                  'theres', 'arent', 'wouldnt', 'hes', 'shes', 'aint', 'alot',  'ppl', 'didnt', 'doesnt', ]

ex_stop_words = stop_words + new_stop_words

# function to filter for Unigrams
def filterUni(ngrams):
    a= []
    type = ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    tags = nltk.pos_tag(ngrams)
    for (word, pos) in tags:
        if word not in ex_stop_words and len(word) > 2:
            if pos in type:
                a.append(word)
    return a

# filter unigrams
filtered_Uni = filterUni(Unigrams)
print('Filtered Unigrams:\n', filtered_Uni)
print('Number of Filtered Unigrams:\n', len(filtered_Uni))


ngrams = filtered_Uni #np.hstack((filtered_Uni, filtered_bi, filtered_tri))
print('Ngrams:\n', ngrams)
print('Number of Filtered Ngrams:\n', len(ngrams))

# contaganate ngrams into one string
ngrams = list(map(lambda x: x.replace(" ", '_'), ngrams))


# tokenize, remove puntrations, and unnecessary characters
def sent_to_words(sentences):
   for sentence in sentences:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


data_words = list(sent_to_words(df.tweetText))

# Create Dictionary
id2word = corpora.Dictionary(x.split() for x in ngrams)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# get the optimal coherence score
def compute_coherence_values(corpus, dictionary, k):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True,
                                           #alpha=a,
                                           #eta=b
                                                )

    coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, texts=data_words, dictionary=id2word,
                                         coherence='u_mass')
    return coherence_model_lda.get_coherence()

# iterate over the range of topics
# Topics range
min_topics = 30
max_topics = 60
step_size = 1
topics_range = range(min_topics, max_topics, step_size)


# Validation sets
num_of_docs = len(corpus)
model_results = {
                 'Topics': [],
                 'Coherence': []
                 }

conn = sqlite3.connect('LDA_Gridsearch.db')

# show progress
if 1 == 1:
    pbar = tqdm.tqdm(total=30)


    # iterate through number of topics
    for k in topics_range:
        # get the coherence score for the given parameters
        cv = compute_coherence_values(corpus=corpus, dictionary=id2word,
                                                  k=k)
        # Save the model results
        model_results['Topics'].append(k)
        model_results['Coherence'].append(cv)

        pbar.update(1)
        print(model_results)
        pd.DataFrame(model_results).to_sql('lda_results_1', con=conn, if_exists='replace', index=False)

    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()

