import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import nltk
import sqlite3
from sklearn.externals import joblib
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()



# find highest topic coherence score from gridsearch
path1 = '//Users//chulinchen//Documents//turnover - twitter analysis/data/Round 2 Data//LDA_Gridsearch.db' # Round 2 Data//Training data.db' # Quitters//original:Final Cleaned Subjects.db
conn = sqlite3.connect(path1, isolation_level = None)


df1 = pd.read_sql("SELECT Topics, Alpha, Beta, Coherence FROM lda_results_1", conn)
df2 = pd.read_sql("SELECT Topics, Alpha, Beta, Coherence FROM lda_results_2", conn)
df3 = pd.read_sql("SELECT Topics, Alpha, Beta, Coherence FROM lda_results_3", conn)

co_list = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
co_list= co_list.sort_values('Coherence', ascending=False).drop_duplicates(['Topics','Alpha', 'Beta'])
pd.set_option('display.max_colwidth', None)
print(co_list.head(10))

co_list.to_excel("//Users//chulinchen//Documents//turnover - twitter analysis/data/Round 2 Data//coherence_rank.xlsx", index=False)


# connect to db
path = '//Users//chulinchen//Documents//turnover - twitter analysis//data//Quitters//Clean 2Class Subjects(-stay).db' # Round 2 Data//Training data.db' # Quitters//original:Final Cleaned Subjects.db
conn = sqlite3.connect(path, isolation_level = None)

# read LIMIT n rows starting from OFFSET m in the large file with specified chunksize
df = pd.read_sql("SELECT username, tweetText, label FROM turnover_tweets", conn)
pd.set_option('display.max_colwidth', None)

# training data
#df = df.iloc[0:12826, :]


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

# function to filter for ADJ + NN/Verb, Verb + NN/Verb, NN + NN/Verb bigrams
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


ngrams = filtered_Uni #np.hstack((filtered_Uni, filtered_bi, filtered_tri))
print('Ngrams:\n', ngrams)
print('Number of Filtered Ngrams:\n', len(ngrams))

# contaganate ngrams into one string
ngrams = list(map(lambda x: x.replace(" ", '_'), ngrams))


# tokenize, remove punctuations, and unnecessary characters
def sent_to_words(sentences):
   for sentence in sentences:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# remove stopword on the extended list
#def remove_stopwords(texts):
#    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


data_words = list(sent_to_words(df.tweetText))
#data_words_nostops = remove_stopwords(data_words)
#print(data_words_nostops)

# Create Dictionary
id2word = corpora.Dictionary(x.split() for x in ngrams)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])

# Human readable format of corpus (term-frequency) for document 1
#print([[(id2word[id], freq) for id, freq in cp if freq > 9] for cp in corpus[:1]])


# Build LDA model
n_topics = 62

lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=n_topics,
                                           random_state=100,
                                           #update_every=1,
                                           chunksize=100,
                                           passes=10,
                                          # alpha='asymmetric',
                                           #eta = 0.91,
                                           per_word_topics=True,
                                                    workers=4)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=n_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            per_word_topics=True,
                                            # alpha=a,
                                            # eta=b
                                            )


lda_model = gensim.models.LdaModel.load('gensim_LDA_62')

# save model to disk (no need to use pickle module)
#lda_model.save('gensim_LDA_42')
#lda_model = LdaMulticore.load('gensim_LDA')

# Print the Keyword in topics
pprint(lda_model.print_topics(num_words=30, num_topics=-1))

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence= 'c_v', processes = -1)
coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, texts=data_words, dictionary=id2word, coherence= 'u_mass', processes = -1)
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda) # for u mass, the closer to zero the better


# print topic-document associations and count
all_topics = lda_model.get_document_topics(corpus, minimum_probability=0)
num_docs = len(all_topics)
all_topics_csr = gensim.matutils.corpus2csc(all_topics)
all_topics_numpy = all_topics_csr.T.toarray()


docsVStopics = pd.DataFrame(all_topics_numpy, columns=["Topic"+str(i) for i in range(n_topics)])
print("Created a (%dx%d) document-topic matrix." % (docsVStopics.shape[0], docsVStopics.shape[1]))
print(docsVStopics)
joblib.dump(docsVStopics, 'Gensim LDA matrix_final(n=62)_whole.pkl')



# Visualization
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, n_jobs= 61, sort_topics=False)
pyLDAvis.display(vis)
pyLDAvis.save_html(vis, 'lda_62.html')
vis
