import spacy
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from multiprocessing import Pool

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
import textblob
from textblob import TextBlob

nlp=spacy.load('en')

# My list of stop words.
stop_list = ["Mrs.", "Ms.", "say", "WASHINGTON", "'s", "Mr.", ]

# Updates spaCy's default stop words list with my additional words.
nlp.Defaults.stop_words.update(stop_list)

# Iterates over the words in the stop words list and resets the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]

def spell_check(doc):
    doc=[TextBlob(token).correct().words[0] for token in doc]
    doc= ' '.join(doc)
    return nlp.make_doc(doc)


def lemmatizer(doc):
    # This takes in a doc of tokens from the NER and lemmatizes them.
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)


def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop !=
           True and token.is_punct != True]
    return doc

# The add_pipe function appends our functions to the default pipeline.
#nlp.add_pipe(spell_check,name='spell_check',after='tagger')
nlp.add_pipe(lemmatizer,name='lemmatizer',after='ner')
nlp.add_pipe(remove_stopwords, name="stopwords", last=True)

docs=pd.read_csv('Extracted_Blogs.csv',sep=',')
list_docs=docs['text'].values
final_list=np.array_split(list_docs,80)
final_list=[list(x) for x in final_list]
print('Final List made')


from multiprocessing import Pool


def transform(texts):
    print('Working')
    results=list(nlp.pipe(texts))
    print('Done')
    return results


print('Multiprocessing started')
p = Pool(80)
results=p.map(transform,final_list)
print('Multiprocessing ended')
final_doc_list = [item for sublist in results for item in sublist]


# Creates, which is a mapping of word IDs to words.
words = corpora.Dictionary(final_doc_list)

# Turns each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in final_doc_list]
print('Building LDA Model')
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=words,num_topics=10,random_state=2,update_every=1,passes=10,alpha='auto',per_word_topics=True)
print('Building LDA Model Done')
import time 
a=time.time()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word,n_jobs=-1)
b=time.time()
print('Visualizations Done',b-a)
pyLDAvis.save_html(vis, '/home/mith/lda_new.html')
print('Done')


for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(str(i)+": " + topic)
    print()
