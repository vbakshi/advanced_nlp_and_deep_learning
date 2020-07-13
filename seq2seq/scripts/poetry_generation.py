# language modelling - poetry generation 

from __future__ import print_function, division
from builtins import range 

import os
import sys 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences 
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, Input, GlobalMaxPooling1D, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model 
from keras.optimizers import Adam, SGD 
from sklearn.metrics import roc_auc_score 

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIMENSION = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM = 25 

#load data 

input_texts = []
target_texts = []

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("../data/robert_frost.txt", encoding='utf-8') as f:
    for line in f.readlines():
        line = line.rstrip()
        if line:
            input_line = '<sos> ' + line 
            output_line = line + ' <eos>'
            input_texts.append(input_line)
            target_texts.append(output_line)

all_lines = input_texts + target_texts 

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# find max seq length
max_seq_length_from_data = max([len(s) for s in input_sequences])
print("Max sequence length from data: {}".format(max_seq_length_from_data))

# get word to index mapping

word2idx = tokenizer.word_index
print("found {} unique tokens".format(len(word2idx)))

# pad sequences to get N x T matrix 
max_sequence_length = min(max_seq_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
print('Shape of data tensor: ', input_sequences.shape)

# load pre-trained word vectors

print("Loading word vectors ...")
word2Vec = {}

with open("../../embeddings/glove.6B/glove.6B.{}d.txt".format(EMBEDDING_DIMENSION), encoding='utf-8') as f:

    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asmatrix(values[1:], dtype='float32')
        word2Vec[word] = vec 
    
print("found {} word vectors".format(len(word2Vec))) 

#prepare embedding matrix 
print("Creating embedding matrix using pretrained vectors ...")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSION), dtype='float32')

for k,v in word2idx.items():

    if v <= MAX_VOCAB_SIZE:
        embedding_vector = word2Vec.get(k)

        if embedding_vector is not None:
            embedding_matrix[v] = embedding_vector

print("shape of embedding matrix".format(embedding_matrix.shape))

# one hot encode the target labels

one_hot_targets = np.zeros((len(target_sequences), max_sequence_length, num_words ))

for i, target_sequence in enumerate(target_sequences):
    for t, word in enumerate(target_sequence):
        if word >0:
            one_hot_targets[i,t,word] = 1

# load pre-trained embeddings into an Embedding layer

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIMENSION,
    weights = [embedding_matrix]
)

print("Building model ...")

# create LSTM layer

input_ = Input(shape = (max_sequence_length,))
initial_h = Input(shape = (LATENT_DIM,))
initial_c = Input(shape = (LATENT_DIM,))

x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
x, _, _ = lstm(x, initial_state = [initial_h, initial_c])
dense = Dense(num_words, activation='softmax')
output = dense(x)

model = Model([input_, initial_h, initial_c], output)

model.compile(
    loss='categorical_crossentropy',
    optimizer = Adam(lr=0.01),
    metrics = ['accuracy']
)

print("Training model ...")

z = np.zeros((len(input_sequences), LATENT_DIM))
r = model.fit(
    [input_sequences, z, z],
    one_hot_targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig("../output/loss_by_epoch.png")


# making a sampling model

input2 = Input(shape=(1,))
x = embedding_layer(input2)
x,h,c = lstm(x, [initial_h, initial_c])
output2 = dense(x)
sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])

idx2Word = {v:k for k,v in word2idx.items()}

def sample_line():

    np_input = np.array([[word2idx['<sos>']]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))

    eos = word2idx['<eos>']

    output_sentence = []

    for _ in range(max_sequence_length):
        o,h,c = sampling_model.predict([np_input, h, c])
        probs = o[0,0]

        if np.argmax(probs) == 0:
            print("first token having max probability - not good")
        
        probs[0] = 0
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p = probs)

        if idx == eos:
            break 
            
        output_sentence.append(idx2Word.get(idx, '<WTF {}>'.format(idx)))

        np_input[0,0] = idx 

    return ' '.join(output_sentence)

with open("../output/generated_poem.txt", 'w') as fh:
    for n in range(4):
        for m in range(4):
            fh.write(sample_line() + '\n')
        fh.write('\n')

        


