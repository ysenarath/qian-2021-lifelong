from pathlib import Path
from typing import List, Tuple
from sklearn.utils import shuffle
import random
import numpy as np

from tokenizers.implementations import BertWordPieceTokenizer

from src.preprocess import tweet_preprocessor

__all__ = [
    "bert_prepare_data",
    "bert_prepare_esoinn_data",
    "bert_prepare_memory",
]

resource_path = Path(__file__).parent.parent / "resources"


def bert_prepare_data(
    data: List[Tuple[str, str, List[str]]],  # [text, pos_group, [neg_groups]]
    max_seq_length: int,
    lower_case: bool = True,
    do_shuffle: bool = True,
    mean_logv_size: int = 0,
):
    data_text = []
    data_pos = []
    data_neg = []
    data_mean = []
    data_logv = []
    if lower_case is True:
        tokenizer = BertWordPieceTokenizer(
            str(resource_path / "bert-base-uncased-vocab.txt"),
            lowercase=True,
        )
    else:
        tokenizer = BertWordPieceTokenizer(
            str(resource_path / "bert-base-cased-vocab.txt"),
            lowercase=False,
        )
    for text, pos_group, neg_groups in data:
        tokens_text = tokenizer.encode(
            tweet_preprocessor(text, lowercase=lower_case)
        ).ids
        if len(tokens_text) <= max_seq_length:
            tokens_text += [0] * (max_seq_length - len(tokens_text))
        else:
            tokens_text = tokens_text[: (max_seq_length - 1)] + [102]
        tokens_pos = tokenizer.encode(pos_group).ids
        if len(tokens_pos) <= max_seq_length:
            tokens_pos += [0] * (max_seq_length - len(tokens_pos))
        else:
            tokens_pos = tokens_pos[: (max_seq_length - 1)] + [102]
        for neg_group in neg_groups:
            tokens_neg = tokenizer.encode(neg_group).ids
            if len(tokens_neg) <= max_seq_length:
                tokens_neg += [0] * (max_seq_length - len(tokens_neg))
            else:
                tokens_neg = tokens_neg[: (max_seq_length - 1)] + [102]
            data_text.append(tokens_text)
            data_pos.append(tokens_pos)
            data_neg.append(tokens_neg)
            if mean_logv_size > 0:
                data_mean.append(np.zeros(mean_logv_size, dtype=np.float32))
                data_logv.append(np.zeros(mean_logv_size, dtype=np.float32))
    if do_shuffle:
        if mean_logv_size == 0:
            shuffled_data_text, shuffled_data_pos, shuffled_data_neg = shuffle(
                data_text, data_pos, data_neg
            )
            return shuffled_data_text, shuffled_data_pos, shuffled_data_neg
        else:
            (
                shuffled_data_text,
                shuffled_data_pos,
                shuffled_data_neg,
                shuffled_mean,
                shuffled_logv,
            ) = shuffle(data_text, data_pos, data_neg, data_mean, data_logv)
            return (
                shuffled_data_text,
                shuffled_data_pos,
                shuffled_data_neg,
                shuffled_mean,
                shuffled_logv,
            )
    else:
        if mean_logv_size == 0:
            return data_text, data_pos, data_neg
        else:
            return data_text, data_pos, data_neg, data_mean, data_logv


def bert_prepare_esoinn_data(
    data: list,
    max_seq_length: int,
    lower_case: bool = True,
    do_shuffle: bool = True,
    return_token_label=False,
):
    data_text = []
    data_pos = []
    data_token_pos = []
    if lower_case is True:
        tokenizer = BertWordPieceTokenizer(
            str(resource_path / "bert-base-uncased-vocab.txt"), lowercase=True
        )
    else:
        tokenizer = BertWordPieceTokenizer(
            str(resource_path / "bert-base-cased-vocab.txt"), lowercase=False
        )
    for text, pos_group, _ in data:
        tokens_text = tokenizer.encode(
            tweet_preprocessor(text, lowercase=lower_case)
        ).ids
        if len(tokens_text) <= max_seq_length:
            tokens_text += [0] * (max_seq_length - len(tokens_text))
        else:
            tokens_text = tokens_text[: (max_seq_length - 1)] + [102]
        if return_token_label:
            tokens_pos = tokenizer.encode(pos_group).ids
            if len(tokens_pos) <= max_seq_length:
                tokens_pos += [0] * (max_seq_length - len(tokens_pos))
            else:
                tokens_pos = tokens_pos[: (max_seq_length - 1)] + [102]
            data_token_pos.append(tokens_pos)
        data_text.append(tokens_text)
        data_pos.append(pos_group)

    if do_shuffle:
        if return_token_label:
            shuffled_data_text, shuffled_data_pos, shuffled_data_token_pos = shuffle(
                data_text, data_pos, data_token_pos
            )
            return shuffled_data_text, shuffled_data_pos, shuffled_data_token_pos
        else:
            shuffled_data_text, shuffled_data_pos = shuffle(data_text, data_pos)
            return shuffled_data_text, shuffled_data_pos
    else:
        if return_token_label:
            return data_text, data_pos, data_token_pos
        else:
            return data_text, data_pos


def bert_prepare_memory(
    memory,
    seen_groups: List[str],
    cand_limit: int,
    max_seq_length: int,
    lower_case: bool = True,
    do_shuffle: bool = True,
    mean_logv_size: int = 0,
):
    all_memory_input = []
    all_memory_fea = []
    all_memory_label = []
    all_memory_neg = []
    all_memory_mean = []
    all_memory_logv = []
    if lower_case is True:
        tokenizer = BertWordPieceTokenizer(
            str(resource_path / "bert-base-uncased-vocab.txt"),
            lowercase=True,
        )
    else:
        tokenizer = BertWordPieceTokenizer(
            str(resource_path / "bert-base-cased-vocab.txt"),
            lowercase=False,
        )
    if mean_logv_size == 0:
        for i, (text_tokens, fea, pos_tokens, pos) in enumerate(memory.values()):
            if i < 50:
                print(tokenizer.decode(text_tokens))
            neg_cand = [cand for cand in seen_groups if cand != pos]
            neg_samples = random.sample(neg_cand, min(len(neg_cand), cand_limit))
            for neg_group in neg_samples:
                tokens_neg = tokenizer.encode(neg_group).ids
                if len(tokens_neg) <= max_seq_length:
                    tokens_neg += [0] * (max_seq_length - len(tokens_neg))
                else:
                    tokens_neg = tokens_neg[: (max_seq_length - 1)] + [102]
                all_memory_input.append(text_tokens)
                all_memory_fea.append(fea)
                all_memory_label.append(pos_tokens)
                all_memory_neg.append(tokens_neg)
        if not do_shuffle:
            return all_memory_input, all_memory_fea, all_memory_label, all_memory_neg
        else:
            return shuffle(
                all_memory_input, all_memory_fea, all_memory_label, all_memory_neg
            )
    else:
        for i, (text_tokens, mean, logv, pos_tokens, pos) in enumerate(memory.values()):
            if i < 50:
                print(tokenizer.decode(text_tokens))
            neg_cand = [cand for cand in seen_groups if cand != pos]
            neg_samples = random.sample(neg_cand, min(len(neg_cand), cand_limit))
            for neg_group in neg_samples:
                tokens_neg = tokenizer.encode(neg_group).ids
                if len(tokens_neg) <= max_seq_length:
                    tokens_neg += [0] * (max_seq_length - len(tokens_neg))
                else:
                    tokens_neg = tokens_neg[: (max_seq_length - 1)] + [102]
                all_memory_input.append(text_tokens)
                all_memory_mean.append(mean)
                all_memory_logv.append(logv)
                all_memory_label.append(pos_tokens)
                all_memory_neg.append(tokens_neg)
        if not do_shuffle:
            return (
                all_memory_input,
                all_memory_label,
                all_memory_neg,
                all_memory_mean,
                all_memory_logv,
            )
        else:
            return shuffle(
                all_memory_input,
                all_memory_label,
                all_memory_neg,
                all_memory_mean,
                all_memory_logv,
            )
