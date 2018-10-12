# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:57:15 2018

@author: mei


Ekstraksi fitur   FastText
"""

from gensim.models import Word2Vec, KeyedVectors
import pandas as pd 
import numpy as np

import pickle, csv, re

import warnings
warnings.filterwarnings("ignore")



#pre-processing
def pre_processing(tweet_list):
    tweet_clean = []
    for tw in tweet_list:
        clean_str = tw.lower() #lowercase
        clean_str = re.sub(r"(?:\@|https?\://)\S+", " ", clean_str) #buang username dan url
        clean_str = re.sub(r'[^\w\s]',' ',clean_str) #buang punctuation
        rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE) #regex kata yang berulang kaya haiiii
        clean_str = re.sub('\s+', ' ', clean_str) # remove extra space
        clean_str = clean_str.strip() #trim depan belakang
        tweet_clean.append(clean_str)
    return tweet_clean


#membuat sentence vector
def sentence_vector(model, text):
    text_split = text.split(' ')
    matrix = list()
    for t in text_split:
        try:
            m = model.wv[t]
            print('kata ada di vocab')
            matrix.append(m)
        except:
            print('kata tidak ada di vocab')
            listofzeros = [0] * 300
            matrix.append(listofzeros)
    return np.average(matrix, axis=0)


def EkstraksiFT(tweet):
    model = KeyedVectors.load_word2vec_format('/home/riset/text_proc/wiki-news-300d-1M.vec', binary=False) 
    list_kalimat = list()
    i = 0
    for d in tweet:
        print(i)
        vec = sentence_vector(model, d).tolist()
        #vc2 = np.matrix(vec)
        #print(type(vec))
        list_kalimat.append(vec)
        i += 1
	#print("----------------")
    return list_kalimat

    
#membaca data
data = pd.read_csv('/home/riset/text_proc/emotion_tweet.csv', delimiter=':', encoding='Latin-1')

tweet_raw = data['tweet']
tweet_clean = pre_processing(tweet_raw)
target_label = data['class'].tolist()


#Ekstraksi fitur  FastText

print("------Ekstraksi fitur FastText----")
list_ft = EkstraksiFT(tweet_clean)
array_ft = np.array(list_ft)


#save fitur  ke pickle 
with open('fitur_ft.pkl', 'wb') as f:
    pickle.dump(array_ft, f)
#p = np.array([np.array(xi) for xi in matrix_kalimat])
print("Fitur FastText berhasil disimpan")

#save label ke pickle

with open('label.pkl', 'wb') as f:
    pickle.dump(target_label, f)
print("Label berhasil disimpan")
