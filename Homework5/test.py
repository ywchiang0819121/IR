from keras_bert import Tokenizer, load_trained_model_from_checkpoint
import codecs
import glob
from keras.preprocessing import sequence
# from keras_V import *
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras
import os
import multiprocessing as mp
import threading

def get_token_dict(dict_path):
    '''
    :param: dict_path: 是bert模型的vocab.txt文件
    :return:將文件中字進行編碼
    '''
    # 將bert模型中的 字 進行編碼
    # 目的是 喂入模型  的是  這些編碼，不是漢字
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
pretrainedPth = 'uncased_L-12_H-768_A-12'
configPth = os.path.join(pretrainedPth, 'bert_config.json')
checkpointPth = os.path.join(pretrainedPth, 'bert_model.ckpt')
vocabPth = os.path.join(pretrainedPth, 'vocab.txt')
def build_bert_model():
    model = load_trained_model_from_checkpoint(configPth, checkpointPth)
    first = keras.layers.Input(shape=(512,))
    firstA = keras.layers.Input(shape=(512,))
    second = model([first, firstA])
    third = keras.layers.Lambda(lambda x: x[:, 0])(second)
    x = keras.layers.Dropout(0.2)(third)
    p = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model([first, firstA], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(0.01),
        metrics=['accuracy']
    )
    model.summary()
    return model

maxlen_doc=512# 句子的最大長度，padding要用的
maxlen_query = 8
dict_path = 'cased_L-12_H-768_A-12/vocab.txt'
token_dict = get_token_dict(dict_path)
tokenizer = Tokenizer(token_dict)
batch_size = 32

test_query_path = glob.glob('./data/test/query/*')
doc_path = glob.glob('./data/doc/*')
doc_name = test_query_path[0].split('\\')[-1]


print('loading data')
docs = []

for path in tqdm(doc_path):
    docs.append(open(path).read())

# docs = np.array(docs)
model = build_bert_model()
model.summary()
model.load_weights('model.h5')

def output():

    with open("submission.csv", mode='w') as submit_file:
        submit_file.write("Query,RetrievedDocuments\n")
        index = 0
        
        
        for query_name in test_query_path:
            print('batch : ', index)
            submit_file.write(query_name.split('\\')[-1] + ",")
            query = open(query_name).read()
            scores = []
            for i in range(0, len(docs), batch_size):
                print(i / len(docs) * 100, ' %')
                codes = []
                for b in range(min(batch_size, len(docs) - i)):
                    code = tokenizer.encode(first = query, second = docs[i + b], max_len=512)
                    code = sequence.pad_sequences(code,maxlen=512,padding='post',truncating='post')
                    codes.append(code)

                codes = np.array(codes)

                score = model.predict([codes[:,0], codes[:,1]])[:,0]
                scores += list(score)


            ranked_doc_idx = np.argsort(scores)[::-1]
            # print(ranked_doc_idx)
            for idx in ranked_doc_idx:
                doc_name = doc_path[idx].split('\\')[-1]
                submit_file.write(" " + doc_name)
            submit_file.write("\n")
            index += 1

if __name__ == "__main__":
    output()
