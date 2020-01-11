# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:07:28 2019

@author: Admin
"""

##Named Entity Recognition with NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

##Now we apply word tokenization and part-of-speech tagging to the sentence
corpus1.question.dtype
corpus1=pd.DataFrame(stop_question)
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

##O/p we get a list of tuples containing the individual words in the 
##sentence and their associated part-of-speech
sent = preprocess(str(corpus1["question"]))
print(sent) 

##Now we implement noun phrase chunking to identify named entities using 
##a regular expression consisting of rules that indicate how sentences should be chunked
pattern = 'NP: {<DT>?<JJ>*<NN>}'
#pattern = r"""Chunk: {<RB.?><VB.?><NNP>+<NN>?}"""

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)
cs.draw()

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

##With the function nltk.ne_chunk(), 
##we can recognize named entities using a classifier  
ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(str(corpus))))
print(ne_tree)
ne_tree.draw()