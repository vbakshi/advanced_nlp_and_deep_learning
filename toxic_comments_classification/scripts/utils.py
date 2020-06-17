import numpy as np 
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer

def load_word_embeddings(embedding_dimension):
    word2vec = dict()
    with open(os.path.join('../../embeddings/glove.6B/glove.6B.{}d.txt').format(embedding_dimension), encoding='utf8') as f:

        for line in f:
            vals = line.split()
            word = vals[0]
            vec = np.asarray(vals[1:], dtype ='float32')

            word2vec[word] = vec 
        print('Found {} word vectors.'.format(len(word2vec)))
    return word2vec

def load_toxic_comments_data(labels):
    train = pd.read_csv(os.path.join("../data/train.csv"))
    train.columns = train.columns.str.strip().str.replace(' ', '_')
    documents = train['comment_text'].fillna("lorem ipsum").values
    target_labels = train[labels].values 
    print("Found {} toxic comments".format(train.shape[0]))
    return documents, target_labels

def tokenize_data(documents, max_vocab_size):
    tokenizer = Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(documents)
    sequences = tokenizer.texts_to_sequences(documents)
    return sequences, tokenizer.word_index
