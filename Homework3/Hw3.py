import os
import math
import time
import itertools
from numba import jit
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

TOPICNUMBER = 80
MAX_ITER = 10000
THRESH = 1e-4
FOLDIN_THRESH = 1e-8
ALPHA = 0.2
BETA = 0.6

doc_path = './Document/'
collection_path = './Collection.txt'
query_path = './Query/'
BGLM_path = './BGLM.txt'

def normalize(array):
      return array / array.sum(axis=0, keepdims=True)

def EM_trainging(row, col, word_doc, word_topic, topic_doc, isFoldin = False):
    print('EM training...')
    L_last = 0
    thresh = None
    if isFoldin == True:
        thresh = FOLDIN_THRESH
    else:
        thresh = THRESH
    E_rsult = np.zeros((TOPICNUMBER, len(word_doc)), dtype=np.float)

    for i in range(MAX_ITER):
        
@jit(nopython=True)


if __name__ == '__main__':
    print('reading file...')
    doc_file = []
    doc = []
    for doc_name in os.listdir(doc_path):
        doc_file.append(doc_name)
        with open(doc_path + doc_name) as f:
            f.readline()
            f.readline()
            f.readline()
            doc += [f.read()]

    query_file = []
    query = []
    for query_name in os.listdir(query_path):
        query_file.append(query_name)
        with open(query_path +  query_name) as f:
            query.append(f.read().replace("-1","").split())

    BGLM = {}
    with open(BGLM_path) as f:
        for s in f.readlines():
            s = s.split()
            BGLM[s[0]] = float((1 - ALPHA - BETA) * np.exp(np.float(s[1])))

    collection = []
    with open(collection_path) as f:
        for s in f.readlines():
            collection += [s]
    
    vectorizer = CountVectorizer(token_pattern='[0-9]+', min_df = 1)
    word_doc_train = vectorizer.fit_transform(collection).tocoo()
    table_train = vectorizer.vocabulary_

    word_topic = np.random.rand(len(table_train), TOPICNUMBER)
    word_topic = normalize(word_topic)
    topic_doc = np.random.rand(TOPICNUMBER, len(collection))
    topic_doc = normalize(topic_doc)

    _, word_topic = EM_training(word_doc_train.row, word_doc_train.col, word_doc_train.data, word_topic, topic_doc, isFoldin = False)

    word_doc = vectorizer.transform(doc).tocoo()

    topic_doc = np.random.rand(TOPICNUMBER, len(doc))
    topic_doc = normalize(topic_doc)
    word_topic = normalize(word_topic)
    topic_doc, word_topic = EM_training(word_doc.row, word_doc.col, word_doc.data, word_topic, topic_doc, isFoldin = True)

    word_doc = word_doc.toarray()
    word_doc = word_doc / word_doc.sum(axis = -1, keepdims = True)
    word_doc = ALPHA * word_doc
    w_PLSA = BETA * np.dot(word_topic, topic_doc).T

    print("computing output...")
    with open("output.csv", mode='w') as output:
        output.write("Query,RetrievedDocuments\n")
        query_index = 0
        for query_name in query_file:
            output.write(query_name + ",")

            log_scores = np.ones((len(doc)), dtype = np.float)
            path = query_path + query_name
            for word in query[query_index]:
                if word in table_train:
                    wi = table_train[word]
                    log_scores += np.log(word_doc[:, wi] + w_PLSA[:, wi] + BGLM[word])
                else:
                    log_scores += np.log(BGLM[word])                    

            ranked_doc_index = np.argsort(log_scores)[::-1]
            for i in ranked_doc_index:
                doc_name = doc_file[i]
                output.write(" " + doc_name)
            query_index += 1
            output.write("\n")