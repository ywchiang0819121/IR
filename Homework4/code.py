import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_ITER = 20
TOP_K = 8
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.5

doc_path = './data/Document/'
query_path = './data/Query/'
def tfidf(query, doc):
    print("computing tf-idf...")
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
    print("update query...")
    for now_iter in range(MAX_ITER):
        print(now_iter)
        # update query
        rel_doc = rank_result[:, :TOP_K]
        non_rel_doc = rank_result[:, -TOP_K:]
        query = ALPHA * query + BETA * doc[rel_doc].mean(axis=1) - GAMMA * doc[non_rel_doc].mean(axis=1)

        rank_result = getRank(query, doc)

    return rank_result

doc_list = []
doc = []
for doc_name in os.listdir(doc_path):
    doc_list.append(doc_name)
    with open(doc_path + doc_name) as f:
        f.readline()
        f.readline()
        f.readline()
        doc += [f.read().replace("-1","").replace("\n","")]
    
query_list = []
query = []
for query_name in os.listdir(query_path):
    query_list.append(query_name)
    with open(query_path +  query_name) as f:
        query.append(f.read().replace("-1","").replace("\n",""))

query_tfidf, doc_tfidf = tfidf(query, doc)
rank_result = RocchioAlgorithm(query_tfidf, doc_tfidf)


with open('result.csv', mode='w') as f:
    f.write('Query,RetrievedDocuments\n')
    for i in range(len(query_list)):
        rank = ' '.join([doc_list[idx] for idx in rank_result[i]])
        f.write('%s,%s\n' % (query_list[i], rank))


