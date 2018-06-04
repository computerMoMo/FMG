# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs
import numpy as np
import matplotlib.pyplot as plt
import math


def hit_k(ground_truth_list, predict_list, k=5):
    gt_num = min(len(ground_truth_list), k)
    hit_num = 0
    for i in range(gt_num):
        pre_item = predict_list[i]
        if pre_item[0] in ground_truth_list:
            hit_num += 1
    return float(hit_num)/len(ground_truth_list)


def ndcg_k(ground_truth_list, predict_list, k=5):
    gt_num = min(len(ground_truth_list), k)
    temp = 0.0
    for i in range(1, gt_num+1):
        temp += 1./math.log(i+1, 2)
    z_k = 1./temp

    sum_num = 0.0
    for i in range(1, gt_num+1):
        pre_item = predict_list[i-1]
        if pre_item[0] in ground_truth_list:
            sum_num += 1./math.log(i+1, 2)
        else:
            sum_num += 0.

    # print("z_k:", z_k, "sum_num:", sum_num)
    return z_k*sum_num


def temp(x):
    x = "%.3f" % float(x)
    return float(x)


if __name__ == "__main__":
    file_path = "fm_res/predict_res_0.05.txt"
    file_reader = codecs.open(file_path, mode="r", encoding="utf-8")
    data_dict = dict()
    head_line = file_reader.readline()
    line = file_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        if line_list[0] not in data_dict:
            data_dict[line_list[0]] = [(line_list[1], float(line_list[2]), float(line_list[3]))]
        else:
            data_dict[line_list[0]].append((line_list[1], float(line_list[2]), float(line_list[3])))
        line = file_reader.readline()
    file_reader.close()

    print(len(data_dict))

    # resort
    hit_k_res = []
    ndcg_k_res = []
    top_k = 15
    for k, v in data_dict.items():
        ground_truth_list = []
        predict_list = []
        for item in v:
            if item[1] > 0.0:
                ground_truth_list.append(item[0])
            predict_list.append((item[0], item[2]))
        predict_list = sorted(predict_list, key=lambda x: x[1], reverse=True)
        hik_k_item = []
        ndcg_k_item = []
        for t_k in range(1, top_k+1):
            hik_k_item.append(hit_k(ground_truth_list, predict_list, t_k))
            ndcg_k_item.append(ndcg_k(ground_truth_list, predict_list, t_k))
            # print("k:", t_k)
            # print("HR:", hit_k(ground_truth_list, predict_list, t_k))
            # print("NDCG:", ndcg_k(ground_truth_list, predict_list, t_k))
        hit_k_res.append(hik_k_item)
        ndcg_k_res.append(ndcg_k_item)

        # print(ground_truth_list)
        # print(predict_list)
        # print(hik_k_item[9])
        # print(ndcg_k_item[9])



    hit_k_res = np.asarray(hit_k_res, dtype=np.float)
    ndcg_k_res = np.asarray(ndcg_k_res, dtype=np.float)
    print(hit_k_res.shape, ndcg_k_res.shape)

    hit_average = []
    ndcg_average = []
    for i in range(top_k):
        hit_average.append(np.mean(hit_k_res[:, i]))
        ndcg_average.append(np.mean(ndcg_k_res[:, i]))
    print(hit_average)
    print(ndcg_average)

    # y_1 = map(temp, ndcg_average)
    y_1 = hit_average
    y_2 = ndcg_average

    x = [i+1 for i in range(15)]
    plt.figure()
    ax1 = plt.subplot()
    ax1.set_xlim([1, 15])
    # ax1.set_ylim([0.6, 0.8])
    ax1.plot(x, y_2, "r",  marker='o')

    plt.xlabel("K")
    plt.ylabel("NDCG@K")
    plt.title("MovieLens")
    my_x_ticks = np.arange(1, 16, 1)

    plt.xticks(my_x_ticks)

    plt.show()