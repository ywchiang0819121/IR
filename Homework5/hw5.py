import os
import keras
import numpy as np
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs

pretrainedPth = 'uncased_L-12_H-768_A-12'
configPth = os.path.join(pretrainedPth, 'bert_config.json')
checkpointPth = os.path.join(pretrainedPth, 'bert_model.ckpt')
vocabPth = os.path.join(pretrainedPth, 'vocab.txt')

import codecs
token_dict = {}
with codecs.open(vocabPth, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

from keras_bert import load_trained_model_from_checkpoint
model = load_trained_model_from_checkpoint(configPth, checkpointPth)

from keras_bert import Tokenizer
tokenizer = Tokenizer(token_dict)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class batchGen:
    def __init__(self, label, bs = 16, tokenizer = None):
        self.batch_size = batch_size
        self.random = random
        self.ans = label
        self.maxlen_doc=512
        self.maxlen_query = 8
        self.tokenizer = tokenizer
        self.iter_index = np.arange(len(self.ans))
    
    def __len__(self):
        return len(self.ans)
    
    def flow(self):
        '''
        Get a batch of data
        '''
        n = len(self.ans)
        i=0
        while(True):
            batch_doc = []
            batch_query = []
            # batch_doc2 = []
            # batch_query2 = []
            batch_labels = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.iter_index)

                index = self.iter_index[i] # choose a data
                doc, query, label = self.GetData(index)
                while doc == None:
                    i = (i+1) % n
                    index = self.iter_index[i] # choose a data
                    doc, query, label = self.GetData(index)
                doc, query, label =  self.data_preprocessing(doc, query, label)
                doc = keras.preprocessing.sequence.pad_sequences(doc,maxlen=self.maxlen_doc,padding='post',truncating='post')
                query = keras.preprocessing.sequence.pad_sequences(query,maxlen=self.maxlen_query,padding='post',truncating='post')
                batch_doc.append(doc[0])
                batch_query.append(query[0])
                # batch_doc2.append(doc[1])
                # batch_query2.append(query[1])
                batch_labels.append(label)

                i = (i+1) % n

            batch_doc = np.array(batch_doc, dtype = np.float32)
            batch_query = np.array(batch_query, dtype = np.float32)
            # batch_doc2 = np.array(batch_doc2, dtype = np.float32)
            # batch_query2 = np.array(batch_query2, dtype = np.float32)
            batch_labels = np.array(batch_labels, dtype = np.float32)

            # yield [batch_doc,batch_doc2, batch_query, batch_query2], batch_labels
            yield [batch_doc, batch_query], batch_labels
