#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.char_embed_size = 50
        self.droprate = 0.3
        self.char_embedding = nn.Embedding(len(self.vocab.char2id), self.char_embed_size, padding_idx= 0)
        self.cnn = CNN(self.word_embed_size, self.char_embed_size)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(self.droprate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        # print("input ", input.size())
        sentence_length = input.size()[0]
        batch_size      = input.size()[1]
        max_word_length = input.size()[2]
        # print("input: ", input)
        char_emb = self.char_embedding(input) # shape (sentence_length, batch_size, max_word_length, char_embed_size)
        x_reshape = char_emb.permute(0,1,3,2) # shape (sentence_length, batch_size, char_embed_size, max_word_length)
        # print("char emb: ", char_emb.size())
        x_reshape = x_reshape.view(-1, self.char_embed_size, max_word_length)
        # print("x_reshape ", x_reshape.size())
        x_conv_out = self.cnn(x_reshape) # shape (batch_size*sentence_length, word_embed_size)
        # print("x_conv ", x_conv_out.size())
        x_highway = self.highway(x_conv_out)  # shape (batch_size*sentence_length, word_embed_size)
        # print("x_highway ", x_highway.size())
        x_word_emb = self.dropout(x_highway.view(sentence_length, batch_size, self.word_embed_size))
        # print("x_word_emb ", x_word_emb.size())
        return x_word_emb
        ### END YOUR CODE

