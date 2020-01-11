#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from whoosh.fields import Schema, TEXT
from whoosh import index
import os, os.path
from whoosh import index
from whoosh import qparser


df = pd.read_csv('qa_Electronics.csv')
df.columns
#os.chdir("G:/Project NLP")

schema = Schema(question = TEXT (stored = True,  field_boost = 2.0),
                answer = TEXT (stored = True,  field_boost = 2.0),
                text = TEXT)     

# create and populate index
def populate_index(dirname, dataframe, schema):
    # Checks for existing index path and creates one if not present
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    print("Creating the Index")
    ix = index.create_in(dirname, schema)
    with ix.writer() as writer:
        # Imports stories from pandas df
        print("Populating the Index")
        for i in dataframe.index:
            add_stories(i, dataframe, writer)

def add_stories(i, dataframe, writer):   
    writer.update_document(question = str(dataframe.loc[i, "question"]),
                           text = str(dataframe.loc[i, "question"]),
                           answer = str(dataframe.loc[i, "answer"]))
    
            
populate_index("index", df, schema)           

##creates index searcher

def index_search(dirname, search_fields, search_query):
    ix = index.open_dir(dirname)
    schema = ix.schema
    # Create query parser that looks through designated fields in index
    og = qparser.OrGroup.factory(0.9)
    mp = qparser.MultifieldParser(search_fields, schema, group = og)
    # This is the user query
    q = mp.parse(search_query)
    # Actual searcher, prints top 10 hits
    with ix.searcher() as s:
        results = s.search(q, limit = 10)
        print("Search Results: ")
        print(results[0:5])
        for hit in results:
            print("the score",hit.score)
            print("the rank" ,hit.rank)
            print("the document number",hit.docnum)
        
index_search("index", ['answer', 'question'], u"cool")


# In[ ]:


text=index_search
import pickle
#save model to a disk
#filename='C:\Anu\whoosh deployment\finalized_model.sav'
filename = "model.pkl"
pickle.dump(text,open(filename,'wb'))
#loading the model
loaded_model=pickle.load(open(filename,'rb'))
result=loaded_model("index", ['question', 'answer'], u"cool")
print (result)
result = result.to_dict()


# In[ ]:


import numpy as np
import pickle
from flask import Flask, render_template, request          
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,23)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
            #search query 
            query = request.form['QA']
            print(query)
            results = []
            ix = index.open_dir("qadata_Index")
            schema = ix.schema
            # Create query parser that looks through designated fields in index
            og = qparser.OrGroup.factory(0.9)
            mp = qparser.MultifieldParser(['question', 'answer'], schema, group = og)
            # This is the user query
            q = mp.parse(request.form['QA'])
            # Actual searcher, prints top 10 hits
            with ix.searcher() as s:
                results = s.search(q, limit = 5)
                for i in range(5):
                    print(results[i]['question'], str(results[i].score), results[i]['answer'])
                return render_template("result.html",searchquery=request.form['QA'],
                                       Q1=results[0]['question'],A1=results[0]['answer'],
                                       Q2=results[1]['question'],A2=results[1]['answer'],
                                       Q3=results[2]['question'],A3=results[2]['answer'],
                                       Q4=results[3]['question'],A4=results[3]['answer'],
                                       Q5=results[4]['question'],A5=results[4]['answer'])


if __name__=='__main__':
   app.run(debug = True)

