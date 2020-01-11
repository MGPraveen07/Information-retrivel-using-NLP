 import pandas as pd
from whoosh.fields import Schema, TEXT
from whoosh import index
import os, os.path
from whoosh import index
from whoosh import qparser


df = pd.read_csv("C:/Users/Admin/Desktop/ExcelR_projects/qa_Electronics.csv")
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
                           answer = str(dataframe.loc[i, "answer"]),
                           text = str(dataframe.loc[i, "question"]))
    
            
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
            print(hit.frequency)

index_search("Index", ['question', 'answer'], u"battery")

