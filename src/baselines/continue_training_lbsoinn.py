from collections import OrderedDict
import csv, os, random
import numpy as np
import statistics
import torch
from torch import optim
from sklearn.metrics import f1_score, accuracy_score
from typing import *
from collections import OrderedDict

from bilstm_model import BilstmModel
import torch
import torch.nn as nn
from utils import (
    bert_prepare_data,
    bert_prepare_esoinn_data,
    bert_prepare_memory,
)
from sklearn.utils import shuffle
import argparse
from lbsoinn import LBSoinn

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def tolist(tsr):
    return tsr.detach().cpu().tolist()


# def mix_train(model, to_train_data, memory_data,
#           current_dev_data, orginal_dev_data, training_config, model_config):
#     model.train()
#     training_text, training_pos, training_neg = to_train_data
#     mem_text, mem_fea, mem_pos, mem_neg = memory_data
#     training_text+=mem_text
#     training_pos+=mem_pos
#     training_neg+=mem_neg
#     batch_size = training_config['batch_size']
#     gpu = training_config['gpu']
#
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = optim.Adam(parameters, training_config["lr"])
#     prev_loss = 100
#     step=0
#     for epoch in range(training_config['epoch']):
#         model.train()
#         avgloss = 0
#         avgrecoverloss = 0
#         avgklloss = 0
#         avgrankingloss = 0
#         training_text, training_pos, training_neg = shuffle(training_text, training_pos, training_neg)
#         for i in range((len(training_text) - 1) // batch_size + 1):
#             step+=1
#             samples_text = torch.tensor(training_text[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#             if torch.cuda.is_available():
#                 samples_text = samples_text.to(gpu)
#             samples_pos = torch.tensor(training_pos[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#             if torch.cuda.is_available():
#                 samples_pos = samples_pos.to(gpu)
#             samples_neg = torch.tensor(training_neg[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#             if torch.cuda.is_available():
#                 samples_neg = samples_neg.to(gpu)
#             optimizer.zero_grad()
#             pos_score, neg_score, rankingloss, recoverloss, klloss = model(samples_text, samples_pos, samples_neg)
#             loss=training_config['lw1']*rankingloss+recoverloss+(max(1,step/300))*klloss
#             loss.backward()
#             optimizer.step()
#             avgloss += loss.data.item() * samples_text.size(0)
#             avgrecoverloss += recoverloss.item() * samples_text.size(0)
#             avgklloss += klloss.item() * samples_text.size(0)
#             avgrankingloss += rankingloss.item() * samples_text.size(0)
#         dev_pos_score, dev_neg_score = test(model, current_dev_data, training_config['batch_size'],
#                                             training_config['gpu'])
#         eval_dict = evaluate(dev_pos_score, dev_neg_score, orginal_dev_data)
#         avgloss /= len(training_text)
#         avgrankingloss /=len(training_text)
#         avgklloss /=len(training_text)
#         avgrecoverloss /=len(training_text)
#
#         print('avg loss:\t', avgloss,  '\tavg rankingloss:\t', avgrankingloss,
#               '\tavg recoverloss:\t', avgrecoverloss, '\tavg klloss:\t', avgklloss,
#               '\tdev acc:\t', eval_dict['acc'], '\tdev macro:\t', eval_dict['macro_f1'],
#               '\tdev micro:\t', eval_dict['micro_f1'])
#         if prev_loss - avgloss < 5e-3 and avgloss < 0.1:
#             break
#         prev_loss = avgloss
#
#     return model


def mix_train(
    model,
    to_train_data,
    memory_data,
    current_dev_data,
    orginal_dev_data,
    training_config,
    model_config,
):
    model.train()
    training_text, training_pos, training_neg = to_train_data
    mem_text, _, mem_pos, mem_neg = memory_data
    training_text += mem_text
    training_pos += mem_pos
    training_neg += mem_neg
    batch_size = training_config["batch_size"]
    gpu = training_config["gpu"]

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, training_config["lr"])
    prev_loss = 100
    step = 0
    for epoch in range(training_config["epoch"]):
        model.train()
        num_mem = 0
        avgloss = 0

        training_text, training_pos, training_neg = shuffle(
            training_text, training_pos, training_neg
        )
        for i in range((len(training_text) - 1) // batch_size + 1):
            step += 1
            samples_text = torch.tensor(
                training_text[i * batch_size : (i + 1) * batch_size],
                dtype=torch.long,
            )

            samples_pos = torch.tensor(
                training_pos[i * batch_size : (i + 1) * batch_size],
                dtype=torch.long,
            )

            samples_neg = torch.tensor(
                training_neg[i * batch_size : (i + 1) * batch_size],
                dtype=torch.long,
            )

            if torch.cuda.is_available():
                samples_text = samples_text.to(gpu)
                samples_neg = samples_neg.to(gpu)
                samples_pos = samples_pos.to(gpu)

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

        if prev_loss - avgloss < 5e-3 and avgloss < 0.1:
            break
        prev_loss = avgloss

    return model


#
# def train(model, to_train_data, memory_data,
#           current_dev_data, orginal_dev_data, training_config, model_config):
#     model.train()
#     training_text, training_pos, training_neg, training_mean, training_logv = to_train_data
#     mem_text, mem_pos, mem_neg, mem_mean, mem_logv = memory_data
#     training_mask=[0]*len(training_text)+[1]*len(mem_text)
#     training_text+=mem_text
#     training_pos+=mem_pos
#     training_neg+=mem_neg
#     training_logv+=mem_logv
#     training_mean+=mem_mean
#     batch_size = training_config['batch_size']
#     gpu = training_config['gpu']
#
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = optim.Adam(parameters, training_config["lr"])
#     prev_loss = 100
#     step=0
#     for epoch in range(training_config['epoch']):
#         model.train()
#         num_mem = 0
#         avgloss=0
#         avgrecoverloss=0
#         avgklloss=0
#         avgrankingloss=0
#         avgmemklloss=0
#         training_text, training_pos, training_neg, training_mask, training_mean, training_logv = shuffle(training_text, training_pos, training_neg, training_mask, training_mean, training_logv)
#         for i in range((len(training_text) - 1) // batch_size + 1):
#             step+=1
#             samples_text = torch.tensor(training_text[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#
#             samples_pos = torch.tensor(training_pos[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#
#             samples_neg = torch.tensor(training_neg[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#
#             samples_mean = torch.from_numpy(np.asarray(training_mean[i * batch_size:(i + 1) * batch_size]))
#
#             samples_logv = torch.from_numpy(np.asarray(training_logv[i * batch_size:(i + 1) * batch_size]))
#
#             samples_training_mask = torch.tensor(training_mask[i * batch_size:(i+1)*batch_size], dtype=torch.long)
#             if torch.cuda.is_available():
#                 samples_text = samples_text.to(gpu)
#                 samples_neg = samples_neg.to(gpu)
#                 samples_pos = samples_pos.to(gpu)
#                 samples_mean = samples_mean.to(gpu)
#                 samples_logv = samples_logv.to(gpu)
#                 samples_training_mask=samples_training_mask.to(gpu)
#             optimizer.zero_grad()
#             pos_score, neg_socre, rankingloss, recoverloss, klloss, memklloss= model(samples_text, samples_pos,
#                                                                                      samples_neg,
#                                                                                      mem_prior_mean= samples_mean,
#                                                                                      mem_prior_logv=samples_logv,
#                                                                                      training_mask=samples_training_mask)
#             loss=training_config['lw1']*rankingloss + recoverloss + (max(1,step/300))*klloss + memklloss
#
#             loss.backward()
#             optimizer.step()
#             avgloss += loss.data.item() * samples_text.size(0)
#             avgrecoverloss += recoverloss.item() * samples_text.size(0)
#             avgklloss += klloss.item() * samples_text.size(0)
#             avgrankingloss += rankingloss.item() * samples_text.size(0)
#             avgmemklloss += (memklloss* torch.sum(samples_training_mask)).item()
#
#         dev_pos_score, dev_neg_score = test(model, current_dev_data, training_config['batch_size'],
#                                             training_config['gpu'])
#         eval_dict = evaluate(dev_pos_score, dev_neg_score, orginal_dev_data)
#         avgloss /= len(training_text)
#         avgrankingloss /= len(training_text)
#         avgklloss /= len(training_text)
#         avgrecoverloss /= len(training_text)
#         if len(mem_text)>0:
#             avgmemklloss /= len(mem_text)
#
#         print('avg loss:\t', avgloss, '\tavg rankingloss:\t', avgrankingloss,
#               '\tavg recoverloss:\t', avgrecoverloss, '\tavg klloss:\t', avgklloss,
#               '\tavg memklloss:\t', avgmemklloss,
#               '\tdev acc:\t', eval_dict['acc'], '\tdev macro:\t', eval_dict['macro_f1'],
#               '\tdev micro:\t', eval_dict['micro_f1'])
#
#         if prev_loss - avgloss < 5e-3 and avgloss < 0.1:
#             break
#         prev_loss = avgloss
#
#     return model


def get_feature(model, text, batch_size, gpu):
    model.eval()
    feature_list = []
    for i in range((len(text) - 1) // batch_size + 1):
        samples_text = torch.tensor(
            text[i * batch_size : (i + 1) * batch_size], dtype=torch.long
        )
        if torch.cuda.is_available():
            samples_text = samples_text.to(gpu)

        feas = model(samples_text)
        for fea in feas:
            feature_list.append(fea)
    return feature_list


# def get_feature_and_prior_dist(model, text, pos_group, batch_size, gpu):
#     model.eval()
#     feature_list = []
#     mean_list = []
#     logv_list =[]
#     for i in range((len(text) - 1) // batch_size + 1):
#         samples_text = torch.tensor(text[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#         samples_pos = torch.tensor(pos_group[i * batch_size:(i + 1) * batch_size], dtype=torch.long)
#         if torch.cuda.is_available():
#             samples_text = samples_text.to(gpu)
#             samples_pos =samples_pos.to(gpu)
#         feas, means, logvs = model(samples_text, samples_pos, return_pos_dis=True)
#         for fea in feas:
#             feature_list.append(fea)
#         for mean in means:
#             mean_list.append(mean)
#         for logv in logvs:
#             logv_list.append(logv)
#     return feature_list, mean_list, logv_list


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
        pos_score, neg_socre, _ = model(samples_text, samples_pos, samples_neg)
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

    cluster_model = LBSoinn(
        model_config["hidden_size"] * 2 * model_config["num_layers"],
        model_config["max_edge_age"],
        model_config["iter_thres"],
        model_config["c1"],
        model_config["c2"],
        model_config["keep_node"],
        gamma=model_config["gamma"],
    )
    memory = OrderedDict()

    training_data = OrderedDict()
    testing_data = OrderedDict()
    dev_data = OrderedDict()
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

    seen_groups = []

    for i, fname in enumerate(cat_order):
        for text, pos_group, neg_group in training_data[fname[:-4]]:
            if pos_group not in seen_groups:
                seen_groups.append(pos_group)
        current_training_data = remove_unseen_group(
            training_data[fname[:-4]], seen_groups
        )

        to_train_data = current_training_data
        # print(fname, to_train_data[:10])
        wrapped_to_train_data = bert_prepare_data(
            to_train_data,
            training_config["max_seq_length"],
            lower_case=training_config["lower_case"],
            do_shuffle=True,
        )

        current_dev_data = remove_unseen_group(
            dev_data[fname[:-4]], seen_groups
        )
        wrapped_current_dev_data = bert_prepare_data(
            current_dev_data,
            training_config["max_seq_length"],
            lower_case=training_config["lower_case"],
            do_shuffle=False,
        )

        memory_data = bert_prepare_memory(
            memory,
            seen_groups,
            training_config["cand_limit"],
            training_config["max_seq_length"],
            lower_case=training_config["lower_case"],
            do_shuffle=True,
        )

        current_testing_data = []
        forward_testing_data = []
        wrapped_current_testing_data = []
        wrapped_forward_testing_data = []
        wrapped_ranking_current_testing_data = []
        wrapped_ranking_forward_testing_data = []
        for j in range(i + 1):
            current_testing_data.append(
                remove_unseen_group(
                    testing_data[cat_order[j][:-4]], seen_groups
                )
            )
        for j in range(i + 1, len(cat_order) - 1):
            forward_testing_data.append(testing_data[cat_order[j][:-4]])
        for data in current_testing_data:
            wrapped_current_testing_data.append(
                bert_prepare_esoinn_data(
                    data,
                    training_config["max_seq_length"],
                    lower_case=training_config["lower_case"],
                    do_shuffle=False,
                )
            )
            wrapped_ranking_current_testing_data.append(
                bert_prepare_data(
                    data,
                    training_config["max_seq_length"],
                    lower_case=training_config["lower_case"],
                    do_shuffle=False,
                )
            )
        for data in forward_testing_data:
            wrapped_forward_testing_data.append(
                bert_prepare_esoinn_data(
                    data,
                    training_config["max_seq_length"],
                    lower_case=training_config["lower_case"],
                    do_shuffle=False,
                )
            )
            wrapped_ranking_forward_testing_data.append(
                bert_prepare_data(
                    data,
                    training_config["max_seq_length"],
                    lower_case=training_config["lower_case"],
                    do_shuffle=False,
                )
            )

        model = mix_train(
            model,
            wrapped_to_train_data,
            memory_data,
            wrapped_current_dev_data,
            current_dev_data,
            training_config,
            model_config,
        )

        (
            wrapped_to_train_lbsoinn_text,
            to_train_lbsoinn_label,
            to_train_lbsoinn_token_label,
        ) = bert_prepare_esoinn_data(
            to_train_data,
            training_config["max_seq_length"],
            lower_case=training_config["lower_case"],
            do_shuffle=True,
            return_token_label=True,
        )

        train_feas = get_feature(
            model,
            wrapped_to_train_lbsoinn_text,
            training_config["batch_size"],
            training_config["gpu"],
        )
        cluster_model.reset_state()
        for sample_index, train_fea in enumerate(train_feas):
            cluster_model.input_signal(
                train_fea.detach().cpu().numpy(),
                to_train_lbsoinn_label[sample_index],
                (i, sample_index),
            )
        assert len(cluster_model.density) == len(
            cluster_model.nodes_training_index
        )
        sorted_task_sample_index = [
            x
            for x in sorted(
                zip(cluster_model.density, cluster_model.nodes_training_index),
                reverse=True,
            )
        ]
        new_memory = OrderedDict()
        cnt_memory_task = {}
        new_mem_sample_size = int(training_config["memory_size"] / (i + 1))
        for key, value in memory.items():
            if key[0] not in cnt_memory_task:
                cnt_memory_task[key[0]] = 0
            if cnt_memory_task[key[0]] < new_mem_sample_size:
                new_memory[key] = value
                cnt_memory_task[key[0]] += 1
        for density, node_task_sample_index in sorted_task_sample_index:
            if len(new_memory) >= training_config["memory_size"]:
                break
            if len(node_task_sample_index) == 0:
                continue
            for task_index, sample_index in node_task_sample_index:
                if len(new_memory) >= training_config["memory_size"]:
                    break
                if task_index == i:
                    if task_index not in cnt_memory_task:
                        cnt_memory_task[task_index] = 0
                    cnt_memory_task[task_index] += 1
                    new_memory[(task_index, sample_index)] = (
                        wrapped_to_train_lbsoinn_text[sample_index],
                        train_feas[sample_index].detach().cpu().numpy(),
                        to_train_lbsoinn_token_label[sample_index],
                        to_train_lbsoinn_label[sample_index],
                    )

        print("memroy per task size: ", cnt_memory_task)
        print(
            "memory size: ",
            len(new_memory),
            " nodes num: ",
            len(cluster_model.nodes),
        )
        memory = new_memory.copy()
        del new_memory

        preds = [
            test(
                model,
                test_set,
                training_config["batch_size"],
                training_config["gpu"],
            )
            for test_set in wrapped_ranking_current_testing_data
        ]
        average_results = {"acc": [], "macro": [], "micro": []}
        for ind_pred, pred in enumerate(preds):
            pos_scores, neg_scores = pred
            result = evaluate(
                pos_scores, neg_scores, current_testing_data[ind_pred]
            )
            average_results["acc"].append(result["acc"])
            average_results["micro"].append(result["micro_f1"])
            average_results["macro"].append(result["macro_f1"])
            print(
                "ranking back macro:\t",
                result["macro_f1"],
                "\tranking back micro:\t",
                result["micro_f1"],
                "\tranking back acc\t",
                result["acc"],
            )
        print(
            "ranking back avg macro:\t",
            statistics.mean(average_results["macro"]),
            "\tranking back avg micro:\t",
            statistics.mean(average_results["micro"]),
            "\tranking back avg acc:\t",
            statistics.mean(average_results["acc"]),
        )

        if training_config["forward"] and i < len(cat_order) - 1:
            preds = [
                test(
                    model,
                    test_set,
                    training_config["batch_size"],
                    training_config["gpu"],
                )
                for test_set in wrapped_ranking_forward_testing_data
            ]
            average_results = {"acc": [], "macro": [], "micro": []}
            for ind_pred, pred in enumerate(preds):
                pos_scores, neg_scores = pred
                result = evaluate(
                    pos_scores, neg_scores, forward_testing_data[ind_pred]
                )
                average_results["acc"].append(result["acc"])
                average_results["micro"].append(result["micro_f1"])
                average_results["macro"].append(result["macro_f1"])
                print(
                    "ranking forw macro:\t",
                    result["macro_f1"],
                    "\tranking forw micro:\t",
                    result["micro_f1"],
                    "\tranking forw acc\t",
                    result["acc"],
                )
            print(
                "ranking forw avg macro:\t",
                statistics.mean(average_results["macro"]),
                "\tranking forw avg micro:\t",
                statistics.mean(average_results["micro"]),
                "\tranking forw avg acc:\t",
                statistics.mean(average_results["acc"]),
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
    parser.add_argument(
        "--memory_size", default=1000, type=int, help="memory size"
    )
    parser.add_argument(
        "--fea_loss", default="cos", type=str, help="cos or mse"
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
    parser.add_argument(
        "--cand_limit",
        type=int,
        help="should be consisitent with training testing dev path",
    )

    parser.add_argument(
        "--iter_thres",
        default=1000,
        type=int,
        help="lbsoinn iteration threshold",
    )
    parser.add_argument(
        "--max_edge_age", default=50, type=int, help="lbsoinn maximum edge age"
    )
    parser.add_argument("--c1", default=0.001, type=float, help="lbsoinn c1")
    parser.add_argument("--c2", default=1.0, type=float, help="lbsoinn c2")
    parser.add_argument(
        "--gamma", default=1.04, type=float, help="lbsoinn gamma"
    )

    parser.add_argument(
        "--keep_all_node",
        default=False,
        action="store_true",
        help="disable deleting noisy nodes",
    )

    options = parser.parse_args()

    training_config = {
        "save_model": options.save_model,
        "epoch": options.epoch,
        "lr": options.learning_rate,
        "max_seq_length": options.max_seq_length,
        "batch_size": options.batch_size,
        "lower_case": options.lower_case,
        "gpu": options.gpu,
        "training_num_limit": options.training_num_limit,
        "balance": options.balance,
        "forward": options.forward,
        "cand_limit": options.cand_limit,
        "memory_size": options.memory_size,
        "fea_loss": options.fea_loss,
    }
    model_config = {
        "model": options.model,
        "word_dim": options.word_dim,
        "hidden_size": options.hidden_size,
        "num_layers": options.num_layers,
        "dropout_prob": options.dropout,
        "loss_margin": options.loss_margin,
        "similarity": options.similarity,
        "gpu": options.gpu,
        "iter_thres": options.iter_thres,
        "max_edge_age": options.max_edge_age,
        "c1": options.c1,
        "c2": options.c2,
        "keep_node": options.keep_all_node,
        "gamma": options.gamma,
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
    #              'christian_identity.csv']

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
    #            'radical_traditional_catholic.csv']
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
    #              'ku_klux_klan.csv']
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
