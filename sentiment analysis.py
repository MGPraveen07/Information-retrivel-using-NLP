import pandas as pd
import json
import os
import ast
import numpy as np
import nltk
 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

sentence = 'Food is good here.'

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    
sentiment_analyzer_scores(sentence)#--{'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.4404}
sentiments=sentiment_analyzer_scores(str('answer_list'))
df.answer_final.head()

from textblob import TextBlob
Text = df.answer_final
df[['polarity', 'subjectivity']] = df['answer_final'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
df[['polarity', 'subjectivity']].head()
polarity_subjectivity = df['answer_final'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
#---------------

import seaborn as sns
from seaborn import boxenplot
import matplotlib.pyplot as plt

#Distribution of Polarity

num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df.polarity, num_bins, facecolor='red', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of polarity')
plt.show();

num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df.subjectivity, num_bins, facecolor='red', alpha=0.5)
plt.xlabel('subjectivity')
plt.ylabel('Count')
plt.title('Histogram of subjectivity')
plt.show();

