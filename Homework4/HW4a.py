import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_ITER = 20
TOP_K = 8
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.5

docpth = './Document/'
quepth = './Query/'

def tfidf(query, doc):
    vectorizer = TfidfVectorizer(smooth_idf = True, sublinear_tf = True)
    doc_tfidf = vectorizer.fit_transform(doc).toarray()
    query_tfidf = vectorizer.transform(query).toarray()
    return query_tfidf, doc_tfidf

def getRank(query, doc):
    rank_result = np.argsort(cosine_similarity(query, doc), axis = 1)
    rank_result = np.flip(rank_result, axis=1)
    return rank_result

def RocchioAlgorithm(query, doc):
    rank_result = getRank(query, doc)
    for i in range(MAX_ITER):
        rel_doc = rank_result[:, :TOP_K]
        non_rel_doc = rank_result[:, -TOP_K:]
        query = ALPHA * query + BETA * doc[rel_doc].mean(axis=1) - GAMMA * doc[non_rel_doc].mean(axis=1)

        rank_result = getRank(query, doc)

    return rank_result

doclist = [] 
doc = [] 
for i in os.listdir(docpth):
    doclist.append(i)
    with open(docpth + i) as f:
        f.readline()
        f.readline()
        f.readline()
        doc += [f.read().replace("-1","").replace("\n","")]

query_list = []
query = []
for i in os.listdir(quepth):
    query_list.append(i)
    with open(quepth +  i) as f:
        query.append(f.read().replace("-1","").replace("\n",""))

query_tfidf, doc_tfidf = tfidf(query, doc)
rank_result = RocchioAlgorithm(query_tfidf, doc_tfidf)

with open('result.csv', mode='w') as f:
    f.write('Query,RetrievedDocuments\n')
    for i in range(len(query_list)):
        rank = ' '.join([doclist[idx] for idx in rank_result[i]])
        f.write('%s,%s\n' % (query_list[i], rank))
