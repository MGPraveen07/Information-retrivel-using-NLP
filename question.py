import pandas as pd

df=pd.read_csv("G:/Project NLP/qa_Electronics.csv")
list(df.columns)
df.shape
df.info()

##answer##
qes=df.iloc[:,4]
qes=pd.DataFrame(qes)
qes.size

qes["question"].isnull().sum() ##checking Null values
# dropping null value columns to avoid errors
qes.dropna(subset=['question'], inplace=True) 

pd.value_counts(qes['question']).head()
print(df['question'].nunique())

qes.duplicated().sum() ###39368 duplicated rows in question columns
qes['question'][2]

##Cleaning The Data

import nltk
#nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize

# Apply a first round of text cleaning techniques
import re
import string

## Make text lowercase, remove text in square brackets,
## remove punctuation and remove words containing numbers.

def clean_text_round1(text):
 
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)

# Let's take a look at the updated text
data_clean = pd.DataFrame(qes.question.apply(round1))
data_clean

##Apply a second round of cleaning,Get rid of some additional punctuation
##non-sensical text that was missed the first time around
def clean_text_round2(text):

    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)

# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.question.apply(round2))
data_clean

pd.value_counts(data_clean['question']).head()

data_clean.duplicated().sum()##33944 duplicated rows in data_clean data

data=data_clean.drop_duplicates(['question'])

##Remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_qes = data.question.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
stop_qes.dtype

stop_words = pd.DataFrame(stop_qes)
stop_words.question.nunique()
stop_words.duplicated().sum() ###20697
stop_question = stop_words.drop_duplicates(['question'])
stop_question.question.head()

##Lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(str(text))]

lemm_question=stop_question.question.apply(lemmatize_text)
lemm_question.head(20)
lem_qes=pd.DataFrame(lemm_question)

##doing stemming part
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stem_qes = lem_qes.question.apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))
stem_qes.dtype

ste_question = pd.DataFrame(stem_qes)
ste_question.question.nunique()
ste_question.duplicated().sum() ##2917
stem_question = ste_question.drop_duplicates(['question'])

#Creating word Cloud
# Joinining all the reviews into single paragraph 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

cloud = " ".join(stem_question.question)

wordcloud= WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(cloud)

plt.imshow(wordcloud)

##For positive world cloud 
with open("G:/Assignments/Text Mining/Positive Words.txt","r") as pos:
  poswords = pos.read().split("\n")

stemqes=stem_question.question.tolist()###covertin data frame into list

qes_pos = ' '.join([w for w in stemqes if w in poswords])

wordcloud_pos = WordCloud(
                           background_color = 'black',
                           width =1800,
                           height =1400
                           ).generate(str(qes_pos))
plt.imshow(wordcloud_pos)

##For negative word cloud
with open("G:/Assignments/Text Mining/Negative Words.txt","r") as nos:
    negwords = nos.read().split("\n")  

qes_neg =' '.join([w for w in stemqes if w in negwords])

wordcloud_neg = WordCloud(
                           background_color = 'black',
                           width =1800,
                           height =1400
                           ).generate(str(qes_neg))
plt.imshow(wordcloud_neg)

