# advanced_nlp_and_deep_learning

This repository is for implementing concepts on Advanced NLP and Deep learning.


## 1. Seq to Seq Models:

### Poetry Generation:

Trained a Decoder network of LSTM network whi ch is trained on peotry text by the famous poet Robert Frost.

*Model traning phase*:

Poem lines are passed in as input sequences of fixed length padded with an \<eos\> end of sequence tag. The same peom lines are passed in as target sequences with \<sos\> start of sequence tag.
Thereby using teacher forcing to learn the next word using the current word in the sequence.  

*Poetry Generated*:

he's celebrating something strange.
i guess he found he got more out of me
but i own what you say makes my head spin.

son, you do as you're told! you take the timber-
yes, there's something the dead are keeping back.
and swollen tight and buried under snow.
a rock-strewn town that bow, but i lean someone with going to

'under the shelter of the family tree.'
that there's something the dead are keeping back?
they were a man's his father killed for me.
it's with us in the room though. it's the bones.'

the way he did in life once; but this time
that had budded before the boughs were piled
you _can_ come down from everything to nothing.
to see if the town wanted to take over out

### Machine Transalation

Next task is to use an encoder-decoder LSTM to translate langage text. This architechture is an extension to the poetry generation model where the hidden states from the encoder are used as inputs to the decoder along with the translated sentences for training.
