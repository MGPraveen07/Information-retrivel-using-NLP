#import the libraries to import the file and convert the file to dataframe

#Variable is defined for file path
qafile="G:/Project NLP/qa_Electronics.json"

#To get the first and last line of the JSON File
lastline = None
with open(qafile,"r") as f:
    lineList = f.readlines()
    lastline=lineList[-1]

#To change the directory of the file path
import os
os.getcwd()

#To Write the corrected JSON File after doing the below changes
# Converted Single Quotes to Double Quotes
#Combined the dictionaries as a single string as Tuple

import json
import ast
with open(qafile,"r") as f, open("cleanjsonfile.json","w") as g:
    for i,line in enumerate(f,0):
        if i == 0:
            line=ast.literal_eval(line)
            g.write("["+json.dumps(line)+",")
        elif line == lastline:  
            line=ast.literal_eval(line)
            g.write(json.dumps(line)+"]")
        else:
            line=ast.literal_eval(line)
            line = json.dumps(line)+","
            g.write(line)

#Variable is defined for Corrected file path
cleanjsonfile="C:/Users/PC/cleanjsonfile.json"

#Read the Corrected File
with open(cleanjsonfile,"r") as h:
    data2=h.read()

#Converted Corrected JSON File into dataframe
import pandas as pd
df = pd.read_json(data2)
type(df)

#To display first 5 rows
df.head()

#To display last 5 rows
df.tail()

#Convert the dataframe to csv
df.to_csv(r"G:/Project NLP/qa_Electronics.csv", index = None, header=True)
