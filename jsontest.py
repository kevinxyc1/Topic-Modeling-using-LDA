#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:43:17 2020

@author: kev
"""

import json

data = []
with open('AAN_20200206.json') as f:
    for line in f:
        data.append(json.loads(line))

title = ['Hi'] * 1000
abstract = ['Hi'] * 1000
for i in range(1000):
    if abstract[i] != 'null':
        title[i] = data[i]["title"]
        abstract[i] = data[i]["abstract"]
 
#for i in range(1000):
#    print(i+1, title[i])
#    
#        
#for i in range(1000):
#    print(i+1, abstract[i])
    
    
res = {title[i]: abstract[i] for i in range(len(title))} 
final = {k:v for k,v in res.items() if v is not None}

#print ("Resultant dictionary is : " +  str(final)) 

abstract_list=list(final.values())
title_list=list(final.keys())

    
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# loop through document list
for i, j in zip(abstract_list, title_list):
    texts = []
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stop words + ing, s, non
    # filter words with number of letters <3
    # stem tokens
    # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=1000)

    print("Title: " + j)
    print("Topics: ", ldamodel.print_topics(num_topics=1, num_words=4))
    print("\n")


   