import os
import keras
import numpy as np
import random
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
from keras_bert import Tokenizer

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class batchGen:
    def __init__(self, label, bs = 16, token_dict = None):
        self.batch_size = bs
        self.random = random
        self.ans = label
        self.maxlen_doc = 512
        self.tokenizer = Tokenizer(token_dict)
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
            batch_doc2 = []
            batch_query = []
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
                x1, x2 = self.tokenizer.encode(first=query, second=doc, max_len=self.maxlen_doc)
                batch_doc.append(x1)
                batch_doc2.append(x2)
                batch_labels.append(label)
                i = (i+1) % n

            batch_doc = np.array(batch_doc, dtype = np.float32)
            batch_doc2 = np.array(batch_doc2, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype = np.float32)

            yield [batch_doc, batch_doc2], batch_labels


    def GetData(self, index):
        [query_fn,doc_fn], label = self.ans[index]
        doc = open('./doc/'+doc_fn).read()
        query = open('./train/query/' + query_fn).read()
        return  doc, query, int(label)

def load_label():
    import random
    ans = []
    labels = open('./train/Pos.txt').read().split('\n')
    for label in labels:
        q,d,p = label.split(' ')
        ans.append([(q,d), p])
    labels = open('./train/Neg.txt').read().split('\n')
    for label in labels:
        q,d,p = label.split(' ')
        ans.append([(q,d), p])
    random.shuffle(ans)
    n = len(ans)
    s = n // 10 * 9
    return ans[:s], ans[s:]


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
checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='loss', verbose=0, mode='min', period=1, save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 3, verbose=1)

label, val_label = load_label()
train_gen = batchGen(label, bs=8, token_dict=token_dict)
val_gen = batchGen(val_label, bs=8, token_dict=token_dict)

model.fit_generator(generator=train_gen.flow(),
                    validation_data = val_gen.flow(), 
                    validation_steps = max(1, len(val_gen) // 8),
                    steps_per_epoch=max(1, len(train_gen)//8),
                    epochs=999999,
                    verbose=1,
                    callbacks=[checkpoint, reduce_lr]
                    )