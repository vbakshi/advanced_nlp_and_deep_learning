from __future__ import print_function, division
from builtins import range 

import os
import sys 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences 
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional, Embedding, Dropout
from keras.models import Model 
from keras.optimizers import Adam 
from sklearn.metrics import roc_auc_score 
from utils import load_word_embeddings, load_toxic_comments_data, tokenize_data

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIMENSION = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# load in pre-trained embedding
word2vec = load_word_embeddings(EMBEDDING_DIMENSION)

# load toxic comments data
documents, labels = load_toxic_comments_data(label_columns)

print("longest comment: {}".format(max([len(document) for document in documents])))
print("shortest comment: {}".format(min([len(document) for document in documents])))

# convert documents into numerical vectors
sequences, word2idx = tokenize_data(documents, MAX_VOCAB_SIZE)
print("found {} unique tokens".format(len(word2idx)))

# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('shape of data tensor', data.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSION)) # 20,000 X 100

for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words may not be found in externally sources word embeddings
            embedding_matrix[i] = embedding_vector 

embedding_layer  = Embedding(num_words, EMBEDDING_DIMENSION, weights = [embedding_matrix], input_length = MAX_SEQUENCE_LENGTH, trainable=False)

print("Building Model ...")

# train a 1D convnet with global max pooling

input_ = Input(shape = (MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
output = Dense(len(label_columns), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy'] 
    )

print('training model ...')

r = model.fit(
    data,
    labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('../output/loss_by_epochs_rnn.png')

p = model.predict(data)
aucs = [] 

for i in range(6):

    auc = roc_auc_score(labels[:,i], p[:,i])
    aucs.append(auc)

print(np.mean(aucs))