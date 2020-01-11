#Parsing of file
#import the libraries to import the file and convert the file to dataframe
import pandas as pd
import json
import os
import ast
import numpy as np
import nltk
nltk.download()

aa="C:\\Users\\Aditi\\Desktop\\data_science _data _set\\NLP_project\\data_set\\working\\qa_electronics.json"
with open (aa,"r") as f:
    datax = (f.read())
    
datay = datax.replace("\n",",")
data_dict = ast.literal_eval(datay)
df = pd.DataFrame(data_dict,columns=['asin','questionType','answerType','answerTime','unixTime','question',
                   'answer'])
#==========

#EDA :
df = pd.read_csv("C:/Users/Admin/Desktop/ExcelR_projects/qa_Electronics.csv")
#Unique values in each column
df.answerType.value_counts()
df.asin.value_counts()
df.questionType.value_counts()
df.answerTime.value_counts()
df.unixTime.value_counts()
df.question.value_counts()
df.answer.value_counts()

#Calculate how many answer are blank
df['c2'] = np.where(df.question != '',1,0)
df.c2.value_counts()

df['c3'] = np.where(df.answer != '',1,0)
df.c3.value_counts() # in answer column 32 values are blank.

df['question_unanswered'] = np.where(df.answer == '',df.question,0)
df.question_unanswered.value_counts()

#1.Basic Feature Extraction
#1.1 Number of words in answer,question column

df['word_count_answer'] = df['answer'].apply(lambda x: len(str(x).split(" ")))
df[['answer','word_count_answer']].head()

df['word_count_question'] = df['question'].apply(lambda x: len(str(x).split(" ")))
df[['question','word_count_question']].head()

ans_max_len=df.loc[df['word_count_answer'].idxmax()] #answer with maximum length
print(ans_max_len)

ques_max_len=df.loc[df['word_count_question'].idxmax()] #question with maximum length
print(ques_max_len)

df.answerType.value_counts()
df.questionType.value_counts()


#1.2 Number of characters

df['char_count_answer'] = df['answer'].str.len() ## this also includes spaces
df[['answer','char_count_answer']].head()

ans_max_char=df.loc[df['char_count_answer'].idxmax()] #answer with maximum length
print(ans_max_char)

df['char_count_question'] = df['question'].str.len() ## this also includes spaces
df[['question','char_count_question']].head()

ques_max_char=df.loc[df['char_count_question'].idxmax()] #answer with maximum length
print(ques_max_char)

#1.3 Average Word Length
#Here, we simply take the sum of the length of all the words and divide it by
#the total length of the tweet

df['avg_len'] = df['char_count_answer']/df['word_count_answer']
df['avg_len'].head()

#1.4 Number of stopwords
#Generally, while solving an NLP problem, the first thing we do is to
 #remove the stopwords. But sometimes calculating the number of stopwords can 
 #also give us some extra information which we might have been losing before.
 
from nltk.corpus import stopwords # imported stopwords from NLTK, which is a basic NLP library in python.
stop = stopwords.words('english')
df['stopwordsinanswer'] = df['answer'].apply(lambda x: len([x for x in x.split() if x in stop]))
df['stopwordsinanswer_value'] = df['answer'].apply(lambda x:([x for x in x.split() if x in stop]))

df[['answer','stopwordsinanswer']].head()
df[['answer','stopwordsinanswer_value']].head()

df['stopwordsinquestion'] = df['question'].apply(lambda x: len([x for x in x.split() if x in stop]))
df['stopwordsinquestion_value'] = df['question'].apply(lambda x:([x for x in x.split() if x in stop]))

df[['question','stopwordsinanswer']].head()
df[['question','stopwordsinanswer_value']].head()

#1.4Number of special characters
df['answer_hastags'] = df['answer'].apply(lambda x: ([x for x in x.split() if x.startswith('#')]))
df[['answer','answer_hastags']].head()
df.answer_hastags.value_counts()

#1.6 Number of numerics
df['numerics_answer'] = df['answer'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['answer','numerics_answer']].head()

df['numeric_question'] = df['question'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['question','numeric_question']].head()

#1.7 Number of Uppercase words
df['upper_answer'] = df['answer'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['answer','upper_answer']].head()

df['upper_answer_1'] = df['answer'].apply(lambda x: ([x for x in x.split() if x.isupper()]))
df[['answer','upper_answer_1']].head()
df.upper_answer_1.value_counts().sum()

#Taking back-up
df1 = df
#2. Basic Pre-processing
#2.1 Lower case
df['lower_answer'] = df['answer'].apply(lambda x: " ".join(x.lower() for x in x.split())) #conversion in lower case
df[['answer','lower_answer']].head()

df['lower_question'] = df['question'].apply(lambda x: " ".join(x.lower() for x in x.split())) #conversion in lower case
df[['lower_question','question']].head()

#2.2 Removing Punctuation
df['answer_final'] = df['lower_answer'].str.replace('[^\w\s]','')
df[['answer_final','lower_answer']].head()

#2.3 Removal of Stop Words

df['answer_final'] = df['answer_final'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df[['answer_final','lower_answer']].head()

#2.4 Common word removal
freq10 = pd.Series(' '.join(df['answer_final']).split()).value_counts()[:10]# removed word 'camera' as i found it as not good to remove this.
freq10 = freq10.drop(labels=['camera'])

freq20 = pd.Series(' '.join(df['answer_final']).split()).value_counts()[:20]
freq20

freq10 = list(freq10.index)
df['answer_final'] = df['answer_final'].apply(lambda x: " ".join(x for x in x.split() if x not in freq10))
df['answer_final'].head()

#2.5 Rare words removal
freq_rare = pd.Series(' '.join(df['answer_final']).split()).value_counts()[-10:]
freq_rare
freq_rare = list(freq_rare.index)
df['answer_final'] = df['answer_final'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_rare))
df['answer_final'].head()

#2.6 Spelling correction
from textblob import TextBlob
df['answer_final']=df['answer_final'].apply(lambda x: str(TextBlob(x).correct()))
df['answer_final'].head()

#Creating word Cloud
# Joinining all the reviews into single paragraph 
answer_final_string = " ".join(df.answer_final)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud_answer_final_string = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(answer_final_string)

plt.imshow(wordcloud_answer_final_string,interpolation="bilinear")
wordcloud_answer_final_string.to_file("wc_1.jpeg")
os.getcwd()
#2.7 Tokenization
from nltk.tokenize import word_tokenize

df['tokenized_text'] = df['answer_final'].apply(word_tokenize) 
df['tokenized_text'].tail()

#2.8 Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['stemming_col']=df['answer_final'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df['stemming_col'].head()

#2.9 Lemmatization
from textblob import Word
df['answer_final_lemma'] = df['answer_final'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df[['answer_final','answer_final_lemma']].head(30)
df[['answer_final','answer_final_lemma']].tail()
df.to_csv('file1.csv') 

df['answer_final_lemma_on_stem_col'] = df['stemming_col'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df[['answer_final','answer_final_lemma']].head(30)
df[['answer_final','answer_final_lemma']].tail()
df.to_csv('C:\\Users\\Aditi\\Desktop\\data_science _data _set\\NLP_project\\results\\file2.csv') 
