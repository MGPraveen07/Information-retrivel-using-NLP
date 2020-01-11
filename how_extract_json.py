# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:09:00 2019

@author: Admin
"""

import json
import gzip

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

f = open("output.strict", 'w')
for l in parse("qa_Electronics.json.gz"):
  f.write(l + '\n')

import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('qa_Electronics.json.gz')