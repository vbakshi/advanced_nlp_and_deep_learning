# Machine translation

from __future__ import print_function, division
from builtins import range 

import os
import sys 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences 
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, Input, GRU, LSTM
from keras.models import Model 
from keras.utils import to_categorical 
from keras.optimizers import Adam, SGD 
from sklearn.metrics import roc_auc_score 


# config
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256 
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100 
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100 

input_texts = []
target_texts = [] 
target_texts_inputs = []

# load data

t=0

with open("../data/spa.txt", encoding='utf-8') as f:
    for line in f.readlines():
        t+=1

        if t > NUM_SAMPLES:
            break

        if '\t' not in line:
            continue
        
        input_text, translation, _ = line.split('\t')

        target_text = translation + ' <eos>'
        target_text_input = '<sos> '+ translation 

        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)

print("num samples: {}".format(len(input_texts)))
print("example input: {}".format(input_texts[0]))
print("example target: {}".format(target_texts[0]))

# tokenize the inputs 

tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

word2idx_inputs = tokenizer_inputs.word_index 
print("Found {} unique tokens".format(len(word2idx_inputs)))

max_len_input = max([len(s) for s in input_texts])

# tokenize the outputs 

tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts_inputs + target_texts)

target_sequences_input = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)

words2idx_output = tokenizer_outputs.word_index
print("Found {} unique tokens".format(len(words2idx_output)))

num_words_output = len(words2idx_output) + 1

max_len_target = max([len(s) for s in target_sequences])

# pad sequences 

encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder input shape: {}".format(encoder_inputs.shape))
print("sample encoder input: {}".format(encoder_inputs[0]))

decoder_inputs = pad_sequences(target_sequences_input, maxlen=max_len_target, padding='post')
print("decoder input shape: {}".format(decoder_inputs.shape))
print("sample decoder input: {}".format(decoder_inputs[0]))

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# load pre-trained word vectors

print("Loading word vectors ...")
word2Vec = {}

with open("../../embeddings/glove.6B/glove.6B.{}d.txt".format(EMBEDDING_DIM), encoding='utf-8') as f:

    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asmatrix(values[1:], dtype='float32')
        word2Vec[word] = vec 
    
print("found {} word vectors".format(len(word2Vec))) 

#prepare embedding matrix 
print("Creating embedding matrix using pretrained vectors ...")
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM), dtype='float32')

for k,v in word2idx_inputs.items():

    if v <= MAX_NUM_WORDS:
        embedding_vector = word2Vec.get(k)

        if embedding_vector is not None:
            embedding_matrix[v] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights = [embedding_matrix],
    input_length=max_len_input
)

# one hot encode the target labels

decoder_targets_one_hot = np.zeros((len(input_texts), max_len_target, num_words_output), dtype='float32')


for i, target_sequence in enumerate(decoder_targets):
    for t, word in enumerate(target_sequence):
        if word >0:
            decoder_targets_one_hot[i,t,word] = 1

print(decoder_targets_one_hot[0])

######## Building the model ########

encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)

encoder_outputs, h, c = encoder(x)

encoder_states = [h,c]

decoder_input_placeholder = Input(shape=(max_len_target,))

decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_input_placeholder)


decoder_lstm = LSTM(LATENT_DIM, return_state=True, return_sequences=True, dropout=0.5)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder, decoder_input_placeholder], decoder_outputs)

model.compile(
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

r = model.fit(
    [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split=0.2
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("../output/mt_loss_by_epoch.png")




