import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import *
import random

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

from transformers import *


class BertRankModel(nn.Module):
    def __init__(self, kwargs: Dict[str, Any]):
        super(BertRankModel, self).__init__()

        self.MODEL = kwargs["model"]
        self.WORD_DIM = kwargs["word_dim"]
        self.VOCAB_SIZE = kwargs[
            "vocab_size"
        ]  # 30522 for uncased, 28996 for cased
        self.HIDDEN_SIZE = kwargs["hidden_size"]
        self.NUM_LAYERS = kwargs["num_layers"]
        self.DROPOUT_PROB = kwargs["dropout_prob"]
        self.GPU = kwargs["gpu"]
        self.SIM = kwargs["similarity"]
        self.BERT_SHORT = kwargs["bert_short"]

        self.loss = nn.MarginRankingLoss(margin=kwargs["loss_margin"])
        self.cos = nn.CosineSimilarity(dim=1)

        self.encoder = BertModel.from_pretrained(self.BERT_SHORT)
        self.linear_tweet = nn.Linear(
            768, self.HIDDEN_SIZE * self.NUM_LAYERS * 2
        )
        self.linear_group = nn.Linear(
            768, self.HIDDEN_SIZE * self.NUM_LAYERS * 2
        )

    def forward(self, tweet, pos_group=None, neg_group=None, return_fea=False):
        tweet_hidden = self.linear_tweet(
            self.encoder(tweet)[0].detach()[:, 0, :]
        )
        tweet_hidden = F.dropout(
            tweet_hidden, p=self.DROPOUT_PROB, training=self.training
        )
        if pos_group is None:
            return tweet_hidden
        pos_hidden = self.linear_group(
            self.encoder(pos_group)[0].detach()[:, 0, :]
        )
        pos_hidden = F.dropout(
            pos_hidden, p=self.DROPOUT_PROB, training=self.training
        )

        neg_hidden = self.linear_group(
            self.encoder(neg_group)[0].detach()[:, 0, :]
        )
        neg_hidden = F.dropout(
            neg_hidden, p=self.DROPOUT_PROB, training=self.training
        )
        if self.SIM == "cos":
            pos_score = self.cos(tweet_hidden, pos_hidden)
            neg_score = self.cos(tweet_hidden, neg_hidden)
        else:
            assert self.SIM == "l2"
            pos_dist = torch.nn.functional.pairwise_distance(
                tweet_hidden, pos_hidden, p=2
            )
            neg_dist = torch.nn.functional.pairwise_distance(
                tweet_hidden, neg_hidden, p=2
            )
            # norm_value=(torch.exp(-pos_dist)+torch.exp(-neg_dist)).detach().data
            pos_score = torch.exp(-pos_dist)
            neg_score = torch.exp(-neg_dist)
        if torch.cuda.is_available():
            loss = self.loss(
                pos_score,
                neg_score,
                torch.ones(pos_score.size(0)).cuda(self.GPU),
            )
        else:
            loss = self.loss(
                pos_score, neg_score, torch.ones(pos_score.size(0))
            )
        if not return_fea:
            return pos_score, neg_score, loss
        else:
            return (
                pos_score,
                neg_score,
                loss,
                tweet_hidden,
                pos_hidden,
                neg_hidden,
            )
