import pandas as pd
import json
import os
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  

after_lemmatisation =pd.read_csv('D:\\data_science _data _set\\NLP_project\\results\\df_after_lemmatization.csv')

tfidf_ip=after_lemmatisation['tokenized_text'].tolist()

v = TfidfVectorizer()
x = v.fit_transform(after_lemmatisation['tokenized_text'])

after_lemmatisation['tfidf']=list(x)

after_lemmatisation['tfidf_1'] = list(tfidf_1_transformed.toarray())

after_lemmatisation.to_csv('D:\\data_science _data _set\\NLP_project\\results\\tfidf.csv') # df after tfidf

##df['after_lemmatisation'] = list(vectoriser.fit_transform(df['tweets']).toarray()) this will not work

