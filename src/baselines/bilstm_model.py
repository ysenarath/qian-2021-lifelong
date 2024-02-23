
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import *

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


class BilstmModel(nn.Module):
    def __init__(self, kwargs: Dict[str, Any]):
        super(BilstmModel,self).__init__()

        self.MODEL=kwargs["model"]
        self.WORD_DIM = kwargs["word_dim"]
        self.VOCAB_SIZE = kwargs["vocab_size"] #30522 for uncased, 28996 for cased
        self.HIDDEN_SIZE = kwargs["hidden_size"]
        self.NUM_LAYERS=kwargs["num_layers"]
        self.DROPOUT_PROB = kwargs["dropout_prob"]
        self.GPU=kwargs['gpu']
        self.SIM=kwargs['similarity']

        self.loss = nn.MarginRankingLoss(margin=kwargs["loss_margin"])
        self.cos = nn.CosineSimilarity(dim=1)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM, padding_idx=0)
        if self.MODEL == "static" or self.MODEL == "non-static":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
        self.encoder_tweet=nn.LSTM(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_group = nn.LSTM(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, tweet, pos_group=None, neg_group=None, return_fea=False):
        _,(tweet_hidden, _) = self.encoder_tweet(self.embedding(tweet),None)
        tweet_hidden=tweet_hidden.transpose(0,1).contiguous()
        tweet_hidden=F.dropout(tweet_hidden.view(tweet_hidden.size(0),-1),p=self.DROPOUT_PROB,training=self.training)
        if pos_group is None:
            return tweet_hidden
        _, (pos_hidden, _) = self.encoder_group(self.embedding(pos_group), None)
        pos_hidden=pos_hidden.transpose(0,1).contiguous()
        pos_hidden = F.dropout(pos_hidden.view(pos_hidden.size(0),-1), p=self.DROPOUT_PROB, training=self.training)

        _, (neg_hidden, _) = self.encoder_group(self.embedding(neg_group), None)
        neg_hidden=neg_hidden.transpose(0,1).contiguous()
        neg_hidden = F.dropout(neg_hidden.view(neg_hidden.size(0),-1), p=self.DROPOUT_PROB, training=self.training)

        if self.SIM=='cos':
            pos_score = self.cos(tweet_hidden, pos_hidden)
            neg_score = self.cos(tweet_hidden, neg_hidden)
        else:
            assert self.SIM=='l2'
            pos_dist = torch.nn.functional.pairwise_distance(tweet_hidden, pos_hidden,p=2)
            neg_dist = torch.nn.functional.pairwise_distance(tweet_hidden, neg_hidden,p=2)
            # norm_value=(torch.exp(-pos_dist)+torch.exp(-neg_dist)).detach().data
            pos_score= torch.exp(-pos_dist)
            neg_score = torch.exp(-neg_dist)
        if torch.cuda.is_available():
            loss = self.loss(pos_score, neg_score, torch.ones(pos_score.size(0)).cuda(self.GPU))
        else:
            loss=self.loss(pos_score, neg_score, torch.ones(pos_score.size(0)))
        if not return_fea:
            return pos_score, neg_score, loss
        else:
            return pos_score, neg_score, loss, tweet_hidden, pos_hidden, neg_hidden

