# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq


def get_hit_ratio(rank_list, target_item):
    for item in rank_list:
        if item == target_item:
            return 1
    return 0


def get_ndcg(rank_list, target_item):
    for i, item in enumerate(rank_list):
        if item == target_item:
            return math.log(2) / math.log(i + 2)
    return 0


def eval_one_rating(i_gnd, i_pre, K):
    if sum(i_pre) == 0:
        return 0, 0
    map_score = {}
    for item, score in enumerate(i_pre):
        map_score[item] = score

    target_item = i_gnd.index(1.0)

    rank_list = heapq.nlargest(K, map_score, key=map_score.get)
    hit = get_hit_ratio(rank_list, target_item)
    ndcg = get_ndcg(rank_list, target_item)
    return hit, ndcg


if __name__ == "__main__":
    file_path = "fm_res/predict_res_total.txt"
    file_reader = codecs.open(file_path, mode="r", encoding="utf-8")
    count_num = 0
    temp_list = []
    total_hit_res = []
    total_ndcg_res = []

    line = file_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        count_num += 1
        if count_num == 101:
            temp_list.append(line_list)
            # eval
            target_item = ""
            pre_list = []
            ground_list = []
            for item in temp_list:
                pre_list.append(float(item[-1]))
                ground_list.append(float(item[-1]))

            temp_hit = []
            temp_ndcg = []
            for k in range(1, 16):
                hit, ndcg = eval_one_rating(ground_list, pre_list, k)
                temp_hit.append(hit)
                temp_ndcg.append(ndcg)
            total_hit_res.append(temp_hit)
            total_ndcg_res.append(temp_ndcg)

            count_num = 0
            temp_list = []
        else:
            temp_list.append(line_list)

        line = file_reader.readline()
    file_reader.close()

    total_hit_res_array = np.asarray(total_hit_res)
    total_ndcg_res_array = np.asarray(total_ndcg_res)
    print(total_hit_res_array.shape)
    print(total_ndcg_res_array.shape)

    hit_average = []
    ndcg_average = []
    for i in range(15):
        hit_average.append("%.5f" % np.mean(total_hit_res_array[:, i]))
        ndcg_average.append("%.5f" % np.mean(total_ndcg_res_array[:, i]))
    print("hit score:", hit_average)
    print("ndcg score:", ndcg_average)

    score_writer = codecs.open("eval_res.txt", mode="w", encoding="utf-8")
    score_writer.write("\t".join(hit_average) + "\n")
    score_writer.write("\t".join(ndcg_average) + "\n")
    score_writer.close()