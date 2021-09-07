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
path = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Quitters//Clean 2Class Subjects(-stay).db' # Round 2 Data//Training data.db' # Quitters//original:Final Cleaned Subjects.db
conn = sqlite3.connect(path, isolation_level = None)

# read LIMIT n rows starting from OFFSET m in the large file with specified chunksize
df = pd.read_sql("SELECT username, tweetText, label FROM turnover_tweets", conn)
pd.set_option('display.max_colwidth', None)

# remove word len<3
#df['tweetText'] = df['tweetText'].str.replace(r'(\b\w{1,2}\b)', '')

# load ngrams (ngram range = 1 ~ 4)
#a = np.load('ngrams.npz')
#ngrams = a['arr_0']


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
#  'por', 'los', 'pero', 'para', 'como','jai', 'les', 'une', 'mais', 'aku', 'ada', 'estoy', 'cuando','moi',
# 'qui', 'que', 'una', 'suis', 'sur', 'aku', 'toi',
# 'lmao', 'lmaoo', 'lmaooo', 'lmaoooo', 'lmaooooo', 'lmaoooooo', 'lmaooooooo',
# 'lmaoooooooo', 'lmaooooooooo', 'lmaoooooooooo', 'lmaooooooooooo', 'lmaoooooooooooo',
#  'lmaooooooooooooo', 'lmaoooooooooooooo', 'lmaooooooooooooooo','haha', 'hahaha', 'hahah',
# 'lol', 'lool', 'loool', 'looool', 'loooool', 'looooool', 'loooooool', 'looooooool','lolol' , 'omg'
# 'miss', 'eat', 'talk', 'wait', 'watch', 'forget', 'start', 'ask', 'guess', 'send', 'listen', 'know', 'way', 'try',
# 'tomorrow', 'tmr', 'yesterday', 'today', 'year',
# 'gonna', 'wanna', 'gotta', 'tryna', 'pls',  'tbh', 'feel',
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

'''
# load bigrams
b = np.load("//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//Bigrams(128).npz")
Bigrams = b['arr_0']
Bigrams = Bigrams.astype(str)
Bigrams = [' '.join(x) for x in Bigrams]
Bigrams = np.array(Bigrams)
print('Bigrams:\n', Bigrams)
print('Number of Bigrams:\n', len(Bigrams))


# function to filter for ADJ + NN/Verb, Verb + NN/Verb, NN + NN/Verb bigrams
def filterBi(ngrams):
    a = []
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')

    for i in ngrams:
        tags = nltk.pos_tag(i.split())
        if tags[0][0] not in ex_stop_words and tags[1][0] not in ex_stop_words:
            if len(tags[0][0]) > 2 and len(tags[1][0]) > 2:
                if tags[0][1] in first_type and tags[1][1] in second_type:
                    a.append(i)
    return a


# filter bigrams
filtered_bi = filterBi(Bigrams)
print('Filtered Bigrams:\n', filtered_bi)
print('Number of Filtered Bigrams:\n', len(filtered_bi))


# load Trigrams
c = np.load("//Users//chulinchen//Documents//turnover - twitter analysis//data//Round 2 Data//Trigrams(128).npz")
Trigrams = c['arr_0']
Trigrams = Trigrams.astype(str)
Trigrams = [' '.join(x) for x in Trigrams]
Trigrams = np.array(Trigrams)
print('Trigrams:\n', Trigrams)
print('Number of Trigrams:\n', len(Trigrams))

# function to filter for ADJ + NN/Verb, Verb + NN/Verb, NN + NN/Verb triigrams
def filterTri(ngrams):
    a = []
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    third_type = ('NN', 'NNS', 'NNP', 'NNPS','VBD', 'VBG', 'VBN', 'VBP', 'VBZ')

    for i in ngrams:
        tags = nltk.pos_tag(i.split())
        if tags[0][0] not in ex_stop_words and tags[2][0] not in ex_stop_words:
            if len(tags[0][0]) > 1 and len(tags[2][0]) > 1:
                if tags[0][1] in first_type and tags[2][1] in third_type:
                    a.append(i)
    return a


# filter trigrams
filtered_tri = filterTri(Trigrams)
print('Filtered Trigrams:\n', filtered_tri)
print('Number of Filtered Trigrams:\n', len(filtered_tri))
'''

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

    #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence= 'c_v', processes = -1)
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, texts=data_words, dictionary=id2word,
                                         coherence='u_mass')
    return coherence_model_lda.get_coherence()

# iterate over the range of topics, alpha, and beta values

#grid = {}
#grid['Validation_Set'] = {}
# Topics range
min_topics = 30
max_topics = 60
step_size = 1
topics_range = range(min_topics, max_topics, step_size)


# Validation sets
num_of_docs = len(corpus)
#corpus_sets = corpus #[# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
               #gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75),
               #corpus]
#corpus_title = '100% Corpus' #['75% Corpus', '100% Corpus']
model_results = {#'Validation_Set': [],
                 'Topics': [],
                 'Coherence': []
                 }

conn = sqlite3.connect('LDA_2.db')
#c = conn.cursor()
#c.execute('''CREATE TABLE IF NOT EXISTS LDA_results
#    (Topics, Alpha, Beta, Coherence)''')

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=30)


    # iterate through number of topics
    for k in topics_range:


        # get the coherence score for the given parameters
        cv = compute_coherence_values(corpus=corpus, dictionary=id2word,
                                                  k=k)
        # Save the model results
         #model_results['Validation_Set'].append(corpus_title)
        model_results['Topics'].append(k)
        model_results['Coherence'].append(cv)

        pbar.update(1)
        print(model_results)
        #row = [k,a,b,cv]
        #c.execute('insert into LDA_results values (?,?,?,?)', row)
        pd.DataFrame(model_results).to_sql('lda_results_1', con=conn, if_exists='replace', index=False)

    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()

