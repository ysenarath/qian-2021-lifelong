from collections import OrderedDict
import csv, os, random
import numpy as np
import statistics
import torch
from torch import optim
from sklearn.metrics import f1_score, accuracy_score
from typing import *

from bilstm_model import BilstmModel
import torch
from utils import bert_prepare_data
from sklearn.utils import shuffle
import argparse

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def tolist(tsr):
    return tsr.detach().cpu().tolist()


def train(
    model,
    to_train_data,
    current_dev_data,
    current_testing_data,
    orginal_dev_data,
    original_testing_data,
    training_config,
    model_config,
):
    model.train()
    training_text, training_pos, training_neg = to_train_data
    batch_size = training_config["batch_size"]
    gpu = training_config["gpu"]

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, training_config["lr"])

    prev_loss = 100
    for epoch in range(training_config["epoch"]):
        model.train()
        avgloss = 0
        training_text, training_pos, training_neg = shuffle(
            training_text, training_pos, training_neg
        )
        for i in range((len(training_text) - 1) // batch_size + 1):
            # if i%100==99:
            #     print(epoch, i)
            samples_text = torch.tensor(
                training_text[i * batch_size : (i + 1) * batch_size],
                dtype=torch.long,
            )
            if torch.cuda.is_available():
                samples_text = samples_text.to(gpu)
            samples_pos = torch.tensor(
                training_pos[i * batch_size : (i + 1) * batch_size],
                dtype=torch.long,
            )
            if torch.cuda.is_available():
                samples_pos = samples_pos.to(gpu)
            samples_neg = torch.tensor(
                training_neg[i * batch_size : (i + 1) * batch_size],
                dtype=torch.long,
            )
            if torch.cuda.is_available():
                samples_neg = samples_neg.to(gpu)
            optimizer.zero_grad()
            pos_score, neg_socre, loss = model(
                samples_text, samples_pos, samples_neg
            )
            loss.backward()
            optimizer.step()
            avgloss += loss.data.item() * samples_text.size(0)
        dev_pos_score, dev_neg_score = test(
            model,
            current_dev_data,
            training_config["batch_size"],
            training_config["gpu"],
        )
        eval_dict = evaluate(dev_pos_score, dev_neg_score, orginal_dev_data)
        avgloss /= len(training_text)

        print(
            "avg loss:\t",
            avgloss,
            "\tdev acc:\t",
            eval_dict["acc"],
            "\tdev macro:\t",
            eval_dict["macro_f1"],
            "\tdev micro:\t",
            eval_dict["micro_f1"],
        )

        average_results = {"acc": [], "macro": [], "micro": []}
        for testing_data_ind in range(len(current_testing_data)):
            test_pos_score, test_neg_score = test(
                model,
                current_testing_data[testing_data_ind],
                training_config["batch_size"],
                training_config["gpu"],
            )
            eval_dict_test = evaluate(
                test_pos_score,
                test_neg_score,
                original_testing_data[testing_data_ind],
            )
            average_results["acc"].append(eval_dict_test["acc"])
            average_results["micro"].append(eval_dict_test["micro_f1"])
            average_results["macro"].append(eval_dict_test["macro_f1"])
            print(
                "test macro:\t",
                eval_dict_test["macro_f1"],
                "\ttest micro:\t",
                eval_dict_test["micro_f1"],
                "\ttest acc\t",
                eval_dict_test["acc"],
            )
        print(
            "avg macro:\t",
            statistics.mean(average_results["macro"]),
            "\tavg micro:\t",
            statistics.mean(average_results["micro"]),
            "\tavg acc:\t",
            statistics.mean(average_results["acc"]),
        )

        if prev_loss - avgloss < 5e-3 and avgloss < 0.1:
            break
        prev_loss = avgloss

    return model


def test(model, testing_data, batch_size, gpu):
    model.eval()
    testing_text, testing_pos, testing_neg = testing_data
    all_pos_score = []
    all_neg_score = []
    for i in range((len(testing_text) - 1) // batch_size + 1):
        samples_text = torch.tensor(
            testing_text[i * batch_size : (i + 1) * batch_size],
            dtype=torch.long,
        )
        if torch.cuda.is_available():
            samples_text = samples_text.to(gpu)
        samples_pos = torch.tensor(
            testing_pos[i * batch_size : (i + 1) * batch_size],
            dtype=torch.long,
        )
        if torch.cuda.is_available():
            samples_pos = samples_pos.to(gpu)
        samples_neg = torch.tensor(
            testing_neg[i * batch_size : (i + 1) * batch_size],
            dtype=torch.long,
        )
        if torch.cuda.is_available():
            samples_neg = samples_neg.to(gpu)
        pos_score, neg_socre, loss = model(
            samples_text, samples_pos, samples_neg
        )
        all_pos_score += tolist(pos_score)
        all_neg_score += tolist(neg_socre)
    return all_pos_score, all_neg_score


def evaluate(
    pos_scores: list, neg_socres: list, original_data: list
) -> Dict[str, Any]:
    cnt_data_sample = 0
    eval_label = []
    eval_pred = []
    for data_tuple in original_data:
        tweet, pos_group, neg_groups = data_tuple
        max_score_group = None
        max_score = -100
        eval_label.append(pos_group)
        for neg_group in neg_groups:
            pos_score = pos_scores[cnt_data_sample]
            neg_socre = neg_socres[cnt_data_sample]
            if pos_score > neg_socre:
                if pos_score > max_score:
                    max_score = pos_score
                    max_score_group = pos_group
            else:
                if neg_socre > max_score:
                    max_score = neg_socre
                    max_score_group = neg_group
            cnt_data_sample += 1
        eval_pred.append(max_score_group)
    assert cnt_data_sample == len(pos_scores)
    macro_f1 = f1_score(eval_label, eval_pred, average="macro")
    micro_f1 = f1_score(eval_label, eval_pred, average="micro")
    acc = accuracy_score(eval_label, eval_pred)
    dict = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "acc": acc,
    }
    return dict


def remove_unseen_group(dataset, seen_groups):
    cleaned_data = []
    for text, pos_group, neg_group in dataset:
        neg_cands = [cand for cand in neg_group if cand in seen_groups]
        if len(neg_cands) > 0:
            cleaned_data.append((text, pos_group, neg_cands))
        else:
            pass
    return cleaned_data


def random_update_memory(
    memory_data, current_training_data, memory_size, sample_size
):
    if len(current_training_data) < sample_size:
        sample_new_data = [
            random.choice(current_training_data) for _ in range(sample_size)
        ]
    else:
        sample_new_data = random.sample(current_training_data, sample_size)
    sample_old_data = random.sample(memory_data, memory_size - sample_size)
    return sample_new_data + sample_old_data


def run_sequence(
    training_pos_path: str,
    testing_pos_path: str,
    dev_pos_path: str,
    cat_order: list,
    training_config: Dict[str, Any],
    model_config: Dict[str, Any],
):
    if training_config["lower_case"]:
        model_config["vocab_size"] = 30522
    else:
        model_config["vocab_size"] = 28996
    model = BilstmModel(model_config)
    if torch.cuda.is_available():
        model = model.cuda(training_config["gpu"])
    training_data = OrderedDict()
    testing_data = OrderedDict()
    dev_data = OrderedDict()
    seen_groups = []
    for fname in cat_order:
        cat_name = fname[:-4]
        training_data[cat_name] = []
        with open(os.path.join(training_pos_path, fname), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # if len(training_data[cat_name])<training_config['training_num_limit']:
                training_data[cat_name].append(
                    (
                        row["text"],
                        row["pos_group"],
                        row["neg_group"].split("&&"),
                    )
                )

    for key in training_data.keys():
        if training_config["balance"]:
            balanced_data = []
            data_dict = {}
            max_data_cnt = 0
            for text, pos_group, neg_group in training_data[key]:
                if pos_group not in data_dict.keys():
                    data_dict[pos_group] = [0, []]
                data_dict[pos_group][0] += 1
                if data_dict[pos_group][0] > max_data_cnt:
                    max_data_cnt = data_dict[pos_group][0]
                data_dict[pos_group][1].append((text, pos_group, neg_group))
            for group in data_dict.keys():
                balanced_data += data_dict[group][1]
                for rest_cnt in range(max_data_cnt - data_dict[group][0]):
                    balanced_data.append(random.choice(data_dict[group][1]))
            training_data[key] = shuffle(balanced_data)[
                : training_config["training_num_limit"]
            ]
        else:
            training_data[key] = training_data[key][
                : training_config["training_num_limit"]
            ]

    for i, fname in enumerate(cat_order):
        for text, pos_group, neg_group in training_data[fname[:-4]]:
            if pos_group not in seen_groups:
                seen_groups.append(pos_group)

    for fname in cat_order:
        cat_name = fname[:-4]
        testing_data[cat_name] = []
        with open(os.path.join(testing_pos_path, fname), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                testing_data[cat_name].append(
                    (
                        row["text"],
                        row["pos_group"],
                        row["neg_group"].split("&&"),
                    )
                )

    for fname in cat_order:
        cat_name = fname[:-4]
        dev_data[cat_name] = []
        with open(os.path.join(dev_pos_path, fname), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dev_data[cat_name].append(
                    (
                        row["text"],
                        row["pos_group"],
                        row["neg_group"].split("&&"),
                    )
                )

    current_training_data = []
    for value in training_data.values():
        current_training_data += remove_unseen_group(value, seen_groups)
    to_train_data = current_training_data
    # print(fname, to_train_data[:10])
    wrapped_to_train_data = bert_prepare_data(
        to_train_data,
        training_config["max_seq_length"],
        lower_case=training_config["lower_case"],
        do_shuffle=True,
    )

    current_dev_data = []
    for value in dev_data.values():
        current_dev_data += remove_unseen_group(value, seen_groups)
    wrapped_current_dev_data = bert_prepare_data(
        current_dev_data,
        training_config["max_seq_length"],
        lower_case=training_config["lower_case"],
        do_shuffle=False,
    )

    current_testing_data = []
    forward_testing_data = []
    wrapped_current_testing_data = []
    wrapped_forward_testing_data = []
    for j in range(len(cat_order)):
        current_testing_data.append(
            remove_unseen_group(testing_data[cat_order[j][:-4]], seen_groups)
        )

    for data in current_testing_data:
        wrapped_current_testing_data.append(
            bert_prepare_data(
                data,
                training_config["max_seq_length"],
                lower_case=training_config["lower_case"],
                do_shuffle=False,
            )
        )

    model = train(
        model,
        wrapped_to_train_data,
        wrapped_current_dev_data,
        wrapped_current_testing_data,
        current_dev_data,
        current_testing_data,
        training_config,
        model_config,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="-----[rnn-classifier]-----")
    parser.add_argument(
        "--model",
        default="rand",
        help="available models: rand, static, non-static, multichannel",
    )
    parser.add_argument(
        "--save_model",
        default=False,
        action="store_true",
        help="whether saving model or not",
    )
    parser.add_argument(
        "--epoch", default=20, type=int, help="number of max epoch"
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="learning rate"
    )
    parser.add_argument(
        "--hidden_size",
        default=64,
        type=int,
        help="size of hidden layer of lstm",
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="num of lstm layers"
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="the number of gpu to be used"
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="dimension of word embeddings",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="batch size"
    )
    parser.add_argument(
        "--max_seq_length",
        default=40,
        type=int,
        help="maximum number of tokens of the input",
    )
    parser.add_argument(
        "--lower_case",
        default=False,
        action="store_true",
        help="lower case input",
    )
    parser.add_argument(
        "--dropout", default=0.3, type=float, help="dropout probability"
    )
    parser.add_argument(
        "--memory_replay",
        default="NONE",
        help="available replay strategy: NONE, RAND",
    )
    parser.add_argument(
        "--memory_size", default=1000, type=int, help="size of memory"
    )
    parser.add_argument(
        "--loss_margin",
        default=0.5,
        type=float,
        help="margin ranking loss param",
    )
    parser.add_argument(
        "--training_num_limit",
        default=5000,
        type=int,
        help="limit of training instances per task",
    )
    parser.add_argument(
        "--similarity",
        default="cos",
        type=str,
        help="vector similarity metric: cos or l2",
    )

    parser.add_argument("--train_path", type=str, help="path of training data")
    parser.add_argument("--test_path", type=str, help="path of testing data")
    parser.add_argument(
        "--dev_path", type=str, help="path of development data"
    )
    parser.add_argument(
        "--balance",
        default=False,
        action="store_true",
        help="do balanced sampling for training data",
    )
    parser.add_argument(
        "--forward",
        default=False,
        action="store_true",
        help="do forward prediction",
    )

    options = parser.parse_args()

    training_config = {
        "save_model": options.save_model,
        "epoch": options.epoch,
        "lr": options.learning_rate,
        "max_seq_length": options.max_seq_length,
        "batch_size": options.batch_size,
        "lower_case": options.lower_case,
        "memory_replay": options.memory_replay,
        "memory_size": options.memory_size,
        "gpu": options.gpu,
        "training_num_limit": options.training_num_limit,
        "balance": options.balance,
        "forward": options.forward,
    }
    model_config = {
        "model": options.model,
        "word_dim": options.word_dim,
        "hidden_size": options.hidden_size,
        "num_layers": options.num_layers,
        "dropout_prob": options.dropout,
        "loss_margin": options.loss_margin,
        "gpu": options.gpu,
        "similarity": options.similarity,
    }
    cat_order = [
        "anti_immigration.csv",
        "anti_semitism.csv",
        "hate_music.csv",
        "anti_catholic.csv",
        "ku_klux_klan.csv",
        "anti_muslim.csv",
        "black_separatist.csv",
        "white_nationalist.csv",
        "neo_nazi.csv",
        "anti_lgbtq.csv",
        "christian_identity.csv",
        "holocaust_identity.csv",
        "neo_confederate.csv",
        "racist_skinhead.csv",
        "radical_traditional_catholic.csv",
    ]
    # cat_order = ['neo_nazi.csv',
    #              'racist_skinhead.csv',
    #              'anti_semitism.csv',
    #              'ku_klux_klan.csv',
    #              'white_nationalist.csv',
    #              'anti_immigration.csv',
    #              'anti_muslim.csv',
    #              'anti_catholic.csv',
    #              'radical_traditional_catholic.csv',
    #              'anti_lgbtq.csv',
    #              'neo_confederate.csv',
    #              'holocaust_identity.csv',
    #              'hate_music.csv',
    #              'black_separatist.csv',
    #              'christian_identity.csv'
    #              ]

    # cat_order=['ku_klux_klan.csv',
    #            'anti_semitism.csv',
    #            'black_separatist.csv',
    #            'white_nationalist.csv',
    #            'neo_nazi.csv',
    #            'hate_music.csv',
    #            'christian_identity.csv',
    #            'anti_immigration.csv',
    #            'holocaust_identity.csv',
    #            'neo_confederate.csv',
    #            'anti_muslim.csv',
    #            'anti_lgbtq.csv',
    #            'anti_catholic.csv',
    #            'racist_skinhead.csv',
    #            'radical_traditional_catholic.csv'
    #            ]
    # cat_order = ['anti_semitism.csv',
    #              'neo_confederate.csv',
    #              'anti_immigration.csv',
    #              'white_nationalist.csv',
    #              'neo_nazi.csv',
    #              'christian_identity.csv',
    #              'anti_catholic.csv',
    #              'holocaust_identity.csv',
    #              'radical_traditional_catholic.csv',
    #              'anti_lgbtq.csv',
    #              'black_separatist.csv',
    #              'hate_music.csv',
    #              'anti_muslim.csv',
    #              'racist_skinhead.csv',
    #              'ku_klux_klan.csv'
    #              ]
    print("=" * 20 + "INFORMATION" + "=" * 20)
    print(training_config)
    print(model_config)
    print("\t".join(cat_order))
    print(options.train_path)
    print("=" * 20 + "INFORMATION" + "=" * 20)

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    model = run_sequence(
        options.train_path,
        options.test_path,
        options.dev_path,
        cat_order,
        training_config,
        model_config,
    )
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)


if __name__ == "__main__":
    main()
