import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal
import random

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

__all__ = [
    "VAEModel",
    "VAEMaskDecodeModel",
]


class VAEModel(nn.Module):
    def __init__(
        self,
        model: Literal["static", "non-static"],
        word_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_prob: float,
        gpu: str,
        similarity: Literal["cos", "l2"],
        loss_margin: float,
        wv_matrix: np.ndarray,
    ):
        super().__init__()
        self.MODEL = model
        self.WORD_DIM = word_dim
        self.VOCAB_SIZE = vocab_size  # 30522 for uncased, 28996 for cased
        self.HIDDEN_SIZE = hidden_size
        self.NUM_LAYERS = num_layers
        self.DROPOUT_PROB = dropout_prob
        self.GPU = gpu
        self.SIM = similarity

        self.ranking_loss = nn.MarginRankingLoss(margin=loss_margin)
        self.recover_loss = nn.NLLLoss(ignore_index=0)

        self.cos = nn.CosineSimilarity(dim=1)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM, padding_idx=0)
        if self.MODEL == "static" or self.MODEL == "non-static":
            self.WV_MATRIX = wv_matrix
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
        self.encoder_tweet = nn.LSTM(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_tweet = nn.GRU(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_group = nn.LSTM(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.tweet_mean = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.tweet_logv = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.group_mean = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.group_logv = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.z2hidden = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.decoder_out = nn.Linear(2 * self.HIDDEN_SIZE, self.VOCAB_SIZE)

    def forward(
        self,
        tweet,
        pos_group=None,
        neg_group=None,
        return_fea=False,
        return_pos_dis=False,
        tf=0.5,
        no_decoding=False,
        mem_prior_mean=None,
        mem_prior_logv=None,
        training_mask=None,
    ):
        tweet_embedding = self.embedding(tweet)
        _, (tweet_hidden, _) = self.encoder_tweet(tweet_embedding, None)
        tweet_hidden = tweet_hidden.transpose(0, 1).contiguous()
        tweet_hidden = F.dropout(
            tweet_hidden.view(tweet_hidden.size(0), -1),
            p=self.DROPOUT_PROB,
            training=self.training,
        )
        tweet_mean = self.tweet_mean(tweet_hidden)
        tweet_logv = self.tweet_logv(tweet_hidden)
        tweet_std = torch.exp(0.5 * tweet_logv)
        z = torch.randn(tweet.size(0), self.NUM_LAYERS * 2 * self.HIDDEN_SIZE)
        if torch.cuda.is_available():
            z = z.to(self.GPU)
        z = z * tweet_std + tweet_mean

        if pos_group is None:
            return z

        _, (pos_hidden, _) = self.encoder_group(self.embedding(pos_group), None)

        pos_hidden = pos_hidden.transpose(0, 1).contiguous()

        pos_hidden = F.dropout(
            pos_hidden.view(pos_hidden.size(0), -1),
            p=self.DROPOUT_PROB,
            training=self.training,
        )

        pos_mean = self.group_mean(pos_hidden)
        pos_logv = self.group_logv(pos_hidden)

        pos_std = torch.exp(0.5 * pos_logv)
        u = torch.randn(tweet.size(0), self.NUM_LAYERS * 2 * self.HIDDEN_SIZE)
        if torch.cuda.is_available():
            u = u.to(self.GPU)
        u = u * pos_std + pos_mean

        if return_pos_dis:
            return z, pos_mean, pos_logv

        _, (neg_hidden, _) = self.encoder_group(self.embedding(neg_group), None)
        neg_hidden = neg_hidden.transpose(0, 1).contiguous()
        neg_hidden = F.dropout(
            neg_hidden.view(neg_hidden.size(0), -1),
            p=self.DROPOUT_PROB,
            training=self.training,
        )
        neg_mean = self.group_mean(neg_hidden)
        neg_std = torch.exp(0.5 * self.group_logv(neg_hidden))
        v = torch.randn(tweet.size(0), self.NUM_LAYERS * 2 * self.HIDDEN_SIZE)
        if torch.cuda.is_available():
            v = v.to(self.GPU)
        v = v * neg_std + neg_mean

        if self.SIM == "cos":
            pos_score = self.cos(z, u)
            neg_score = self.cos(z, v)
        else:
            assert self.SIM == "l2"
            pos_dist = torch.nn.functional.pairwise_distance(z, u, p=2)
            neg_dist = torch.nn.functional.pairwise_distance(z, v, p=2)
            pos_score = torch.exp(-pos_dist)
            neg_score = torch.exp(-neg_dist)
        if no_decoding:
            if return_fea:
                return pos_score, neg_score, z, u, v
            else:
                return pos_score, neg_score

        decoder_hidden = self.z2hidden(z)
        decoder_hidden = decoder_hidden.view(
            tweet.size(0), self.NUM_LAYERS * 2, self.HIDDEN_SIZE
        )
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()

        decoder_input = tweet[:, 0]
        decoder_prob = torch.zeros(tweet.size(0), tweet.size(1) - 1, self.VOCAB_SIZE)
        if torch.cuda.is_available():
            decoder_prob = decoder_prob.to(self.GPU)
        for t in range(1, tweet.size(1)):
            decoder_input = self.embedding(decoder_input.unsqueeze(1))
            decoder_output, decoder_hidden = self.decoder_tweet(
                decoder_input, decoder_hidden
            )

            vocab_output = self.decoder_out(decoder_output.squeeze(1))
            decoder_prob[:, t - 1, :] = nn.functional.log_softmax(vocab_output, dim=-1)
            teacher_force = random.random() < tf
            decoder_input = vocab_output.argmax(dim=1)
            decoder_input = tweet[:, t] if teacher_force else decoder_input

        if torch.cuda.is_available():
            rankingloss = self.ranking_loss(
                pos_score, neg_score, torch.ones(pos_score.size(0)).cuda(self.GPU)
            )
        else:
            rankingloss = self.ranking_loss(
                pos_score, neg_score, torch.ones(pos_score.size(0))
            )
        # print(decoder_prob.size(), tweet[:,1:].size())
        recoverloss = self.recover_loss(
            decoder_prob.contiguous().view(
                tweet.size(0) * (tweet.size(1) - 1), self.VOCAB_SIZE
            ),
            tweet[:, 1:].contiguous().view(-1),
        )
        klloss = (
            0.5
            * torch.sum(
                -1
                + pos_logv
                - tweet_logv
                + (tweet_logv.exp() + (tweet_mean - pos_mean).pow(2))
                * (1.0 / pos_logv.exp())
            )
            / tweet.size(0)
        )
        if mem_prior_mean is not None:
            unmaskedkl = (
                -1
                + mem_prior_logv
                - pos_logv
                + (pos_logv.exp() + (pos_mean - mem_prior_mean).pow(2))
                * (1.0 / mem_prior_logv.exp())
            )
            if torch.sum(training_mask).item() > 0:
                memklloss = (
                    0.5
                    * torch.sum(
                        torch.mm(training_mask.unsqueeze(0).float(), unmaskedkl)
                    )
                    / torch.sum(training_mask).float()
                )
            else:
                memklloss = 0
        if not return_fea:
            if mem_prior_mean is not None:
                return pos_score, neg_score, rankingloss, recoverloss, klloss, memklloss
            else:
                return pos_score, neg_score, rankingloss, recoverloss, klloss
        else:
            if mem_prior_mean is not None:
                return (
                    pos_score,
                    neg_score,
                    rankingloss,
                    recoverloss,
                    klloss,
                    memklloss,
                    z,
                    u,
                    v,
                )
            else:
                return pos_score, neg_score, rankingloss, recoverloss, klloss, z, u, v


class VAEMaskDecodeModel(nn.Module):
    def __init__(
        self,
        model: Literal["static", "non-static"],
        word_dim: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_prob: float,
        gpu: str,
        similarity: Literal["cos", "l2"],
        loss_margin: float,
        wv_matrix: np.ndarray,
    ):
        super(VAEMaskDecodeModel, self).__init__()
        self.MODEL = model
        self.WORD_DIM = word_dim
        self.VOCAB_SIZE = vocab_size  # 30522 for uncased, 28996 for cased
        self.HIDDEN_SIZE = hidden_size
        self.NUM_LAYERS = num_layers
        self.DROPOUT_PROB = dropout_prob
        self.GPU = gpu
        self.SIM = similarity
        self.MASK_INDEX = 103
        self.CLS_INDEX = 101
        self.SEP_INDEX = 102
        self.PAD_INDEX = 0

        self.ranking_loss = nn.MarginRankingLoss(margin=loss_margin)
        self.recover_loss = nn.NLLLoss(ignore_index=0)

        self.cos = nn.CosineSimilarity(dim=1)

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM, padding_idx=0)
        if self.MODEL == "static" or self.MODEL == "non-static":
            self.WV_MATRIX = wv_matrix
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
        self.encoder_tweet = nn.LSTM(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_tweet = nn.GRU(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_group = nn.LSTM(
            input_size=self.WORD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.tweet_mean = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.tweet_logv = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.group_mean = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.group_logv = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.z2hidden = nn.Linear(
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
            self.NUM_LAYERS * 2 * self.HIDDEN_SIZE,
        )
        self.decoder_out = nn.Linear(2 * self.HIDDEN_SIZE, self.VOCAB_SIZE)

    def forward(
        self,
        tweet,
        pos_group=None,
        neg_group=None,
        return_fea=False,
        return_pos_dis=False,
        wd=0.15,
        no_decoding=False,
        mem_prior_mean=None,
        mem_prior_logv=None,
        training_mask=None,
    ):
        tweet_embedding = self.embedding(tweet)
        _, (tweet_hidden, _) = self.encoder_tweet(tweet_embedding, None)
        tweet_hidden = tweet_hidden.transpose(0, 1).contiguous()
        tweet_hidden = F.dropout(
            tweet_hidden.view(tweet_hidden.size(0), -1),
            p=self.DROPOUT_PROB,
            training=self.training,
        )
        tweet_mean = self.tweet_mean(tweet_hidden)
        tweet_logv = self.tweet_logv(tweet_hidden)
        tweet_std = torch.exp(0.5 * tweet_logv)
        z = torch.randn(tweet.size(0), self.NUM_LAYERS * 2 * self.HIDDEN_SIZE)
        if torch.cuda.is_available():
            z = z.to(self.GPU)
        z = z * tweet_std + tweet_mean

        if pos_group is None:
            return z

        _, (pos_hidden, _) = self.encoder_group(self.embedding(pos_group), None)

        pos_hidden = pos_hidden.transpose(0, 1).contiguous()

        pos_hidden = F.dropout(
            pos_hidden.view(pos_hidden.size(0), -1),
            p=self.DROPOUT_PROB,
            training=self.training,
        )

        pos_mean = self.group_mean(pos_hidden)
        pos_logv = self.group_logv(pos_hidden)

        pos_std = torch.exp(0.5 * pos_logv)
        u = torch.randn(tweet.size(0), self.NUM_LAYERS * 2 * self.HIDDEN_SIZE)
        if torch.cuda.is_available():
            u = u.to(self.GPU)
        u = u * pos_std + pos_mean

        if return_pos_dis:
            return z, pos_mean, pos_logv

        _, (neg_hidden, _) = self.encoder_group(self.embedding(neg_group), None)
        neg_hidden = neg_hidden.transpose(0, 1).contiguous()
        neg_hidden = F.dropout(
            neg_hidden.view(neg_hidden.size(0), -1),
            p=self.DROPOUT_PROB,
            training=self.training,
        )
        neg_mean = self.group_mean(neg_hidden)
        neg_std = torch.exp(0.5 * self.group_logv(neg_hidden))
        v = torch.randn(tweet.size(0), self.NUM_LAYERS * 2 * self.HIDDEN_SIZE)
        if torch.cuda.is_available():
            v = v.to(self.GPU)
        v = v * neg_std + neg_mean

        if self.SIM == "cos":
            pos_score = self.cos(z, u)
            neg_score = self.cos(z, v)
        else:
            assert self.SIM == "l2"
            pos_dist = torch.nn.functional.pairwise_distance(z, u, p=2)
            neg_dist = torch.nn.functional.pairwise_distance(z, v, p=2)
            pos_score = torch.exp(-pos_dist)
            neg_score = torch.exp(-neg_dist)
        if no_decoding:
            if return_fea:
                return pos_score, neg_score, z, u, v
            else:
                return pos_score, neg_score

        decoder_hidden = self.z2hidden(z)
        decoder_hidden = decoder_hidden.view(
            tweet.size(0), self.NUM_LAYERS * 2, self.HIDDEN_SIZE
        )
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()

        if wd > 0:
            prob = torch.rand(tweet.size())
            if torch.cuda.is_available():
                prob = prob.to(self.GPU)
            prob[
                (tweet.data - self.CLS_INDEX)
                * (tweet.data - self.SEP_INDEX)
                * (tweet.data - self.PAD_INDEX)
                == 0
            ] = 1
            decoder_input = tweet.clone()
            decoder_input[prob < wd] = self.MASK_INDEX
            decoder_input_embedding = self.embedding(decoder_input)
        else:
            decoder_input_embedding = tweet_embedding
        decoder_output, _ = self.decoder_tweet(decoder_input_embedding, decoder_hidden)

        vocab_output = self.decoder_out(decoder_output)
        decoder_prob = nn.functional.log_softmax(vocab_output[:, :-1, :], dim=-1)

        if torch.cuda.is_available():
            rankingloss = self.ranking_loss(
                pos_score,
                neg_score,
                torch.ones(pos_score.size(0)).cuda(self.GPU),
            )
        else:
            rankingloss = self.ranking_loss(
                pos_score,
                neg_score,
                torch.ones(pos_score.size(0)),
            )
        # print(decoder_prob.size(), tweet[:,1:].size())
        recoverloss = self.recover_loss(
            decoder_prob.contiguous().view(
                tweet.size(0) * (tweet.size(1) - 1), self.VOCAB_SIZE
            ),
            tweet[:, 1:].contiguous().view(-1),
        )
        klloss = (
            0.5
            * torch.sum(
                -1
                + pos_logv
                - tweet_logv
                + (tweet_logv.exp() + (tweet_mean - pos_mean).pow(2))
                * (1.0 / pos_logv.exp())
            )
            / tweet.size(0)
        )
        if mem_prior_mean is not None:
            unmaskedkl = (
                -1
                + mem_prior_logv
                - pos_logv
                + (pos_logv.exp() + (pos_mean - mem_prior_mean).pow(2))
                * (1.0 / mem_prior_logv.exp())
            )
            if torch.sum(training_mask).item() > 0:
                memklloss = (
                    0.5
                    * torch.sum(
                        torch.mm(training_mask.unsqueeze(0).float(), unmaskedkl)
                    )
                    / torch.sum(training_mask).float()
                )
            else:
                memklloss = 0
        if not return_fea:
            if mem_prior_mean is not None:
                return pos_score, neg_score, rankingloss, recoverloss, klloss, memklloss
            else:
                return pos_score, neg_score, rankingloss, recoverloss, klloss
        else:
            if mem_prior_mean is not None:
                return (
                    pos_score,
                    neg_score,
                    rankingloss,
                    recoverloss,
                    klloss,
                    memklloss,
                    z,
                    u,
                    v,
                )
            else:
                return pos_score, neg_score, rankingloss, recoverloss, klloss, z, u, v
