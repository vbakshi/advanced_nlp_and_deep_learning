from __future__ import print_function, division
from builtins import range, input 

import os 
from keras.models import Model 
from keras.layers import Input, GRU, LSTM, Bidirectional, GlobalMaxPooling1D, Lambda, Concatenate, Dense 
import keras.backend as K 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def load_mnist_data():
    # see notebook for 
    df = pd.read_csv("../data/train.csv")
    data = df.values
    np.random.shuffle(data) 
    X = data[:,1:].reshape(-1,28,28)
    y = data[:,0]

    return X, y 

X, y = load_mnist_data() 

D = 28
M = 15
N = 10

input_ = Input((D,D))

rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_)
x1 = GlobalMaxPooling1D()(x1)

rnn2 = Bidirectional(LSTM(M, return_sequences=True))

permuter = Lambda(lambda t: K.permute_dimensions(t, pattern=(0,2,1)))

x2 = permuter(input_)
x2 = rnn2(x2)
x2 = GlobalMaxPooling1D()(x2) 

x = Concatenate(axis=1)([x1, x2])

output = Dense(N, activation="softmax")(x)

model = Model(input_, output)

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training model ...")

r = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('../output/loss_by_epochs_bidir_lstm.png')

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.savefig('../output/acc_by_epochs_bidir_lstm.png')




