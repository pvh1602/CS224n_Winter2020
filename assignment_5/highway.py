#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, e_word):
        """
        
        Arguments:
            e_word {int} -- size of word embeding
        """
        super(Highway, self).__init__()
        # print("Creating highway net")
        self.e_word = e_word
        self.h_projection = nn.Linear(self.e_word, self.e_word, bias = True)
        self.h_gate       = nn.Linear(self.e_word, self.e_word, bias = True)

    def forward(self, x_conv):
        """
        
        Arguments:
            x_conv {tensor} -- shape (e_word)
        
        Returns:
            [tensor] -- shape (e_word)
        """
        x_proj = F.relu(self.h_projection(x_conv))
        x_gate = torch.sigmoid(self.h_gate(x_conv))
        x_highway = x_gate*x_proj + (1-x_gate)*x_conv

        return x_highway

    ### END YOUR CODE

