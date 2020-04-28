#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_word, e_char, kernel_size = 5, padding = 1):
        """[summary]
        
        Arguments:
            e_word {int} -- [word embedding size]
            e_char {int} -- [char embedding size]
        
        Keyword Arguments:
            kernel_size {int} -- [description] (default: {5})
            padding {int} -- [description] (default: {1})
        """
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.filter_size = e_word
        self.padding = padding
        self.e_char = e_char
        self.conv1d = nn.Conv1d(self.e_char, self.filter_size,\
                         kernel_size= self.kernel_size, padding= self.padding)  
        # self.maxpool = nn.MaxPool1d(kernel_size= self.conv1d.size()[-1] + self.padding, padding= self.padding)

    def forward(self, x_reshape):
        """[summary]
        
        Arguments:
            x_reshape {tensor} -- [result of padding and embedding lookup process having shape (batch_size, e_char, m_word)]
            e_char is char embedding size
            m_word is the length of longest word 
        
        Returns:
            [tensor] -- [x_conv_out is the result after cnn process having shape (batch_size, e_word)]
            e_word is the word embedding size
        """
        x_conv = self.conv1d(x_reshape) #shape(batch_size, word_embeding_size, m_word - kernel_size+1)
        x_conv_out = torch.max(F.relu(x_conv), dim = 2)[0] #shape (batch_size, word_embeding_size)
        return x_conv_out



    ### END YOUR CODE

