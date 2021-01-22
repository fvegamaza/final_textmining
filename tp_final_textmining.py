# -*- coding: utf-8 -*-
"""
Created on Fri Jun 3 15:56:14 2020

Este modelo se basa en realizar una Extrative Summarization el cual su foco está puesto en intentar resumir un artículo seleccionando palabras que retienen la parte más importante del mismo.
Este método pesa la importancia de las partes de la sentencia y usa las más relevantes para realizar un sumario
Input article → split into sentences → remove stop words → build a similarity matrix → generate rank based on matrix → pick top N sentences for summary.

@author: Franco Vega - Sebastian Calcagno 
"""
import pandas as pd
import re
import os #for change directory
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance


accented_string = 'Málaga'
acented = pd.DataFrame(accented_string)
# accented_string is of type 'unicode'
import unidecode
unaccented_string = unidecode.unidecode(accented_string)


#Archivo en .txt
os.chdir('C:\\Users\\fvegamaza\\111\\Spyder\\textmining\\train\\')
filedata = open("train1.txt",encoding="utf8")
filedata = pd.DataFrame(filedata)
filedata.columns = ["texto"]

def sacar_acentos:
    filedata.apply(lambda x: unidecode.unidecode())
    
    
#Cleaning

filedata["texto"] = filedata["texto"].str.lower()
filedata["texto"] = filedata["texto"].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
filedata["texto"] = filedata["texto"].apply(lambda elem: re.sub("\d+", "", elem))


#Reemplace spaces into nan
df = filedata.copy()
df['texto'].replace('', np.nan, inplace=True)
df['texto'].replace('  ', np.nan, inplace=True)
df['texto'].replace('\t\t', np.nan, inplace=True)
df['texto'].replace(' ', np.nan, inplace=True)

#Drop na
df.dropna(subset=['texto'], inplace=True)


##Similarity matrix
sentences = list(df["texto"])
len(sentences)

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

stop_words = stopwords.words('spanish')

sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
sentence_similarity_martix)
#Order by scores
scores = nx.pagerank(sentence_similarity_graph)

#Sort the rank and pick top sentences
ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    

summarize_text = []
for i in range(10):
      summarize_text.append("".join(ranked_sentence[i][1]))  

print("Summarize Text: \n", ".".join(summarize_text))
 
with open("codigo_final.txt", "w") as output:
    output.write(str(summarize_text))