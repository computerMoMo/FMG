# -*- coding:utf-8 -*-
'''
    generate MF features from the meta-structure similarity
'''
from __future__ import print_function
from mf import MF_BGD as MF
from utils import reverse_map
from logging_util import init_logger

import sys
import time
import logging
import numpy as np
import os

topK = 500
data_dir = "data/movie"


def run(path_str, K=10):
    if path_str in ['ratings_only']:
        use_topK = False
    else:
        use_topK = True

    sim_filename = os.path.join(data_dir, 'sim_res/path_count/%s.res' % path_str)
    if path_str == 'ratings_only':
        sim_filename = os.path.join(data_dir, 'tuples/ratings.txt')
    elif use_topK:
        sim_filename = os.path.join(data_dir, 'sim_res/path_count/%s_top%s.res' % (path_str, topK))

    start_time = time.time()
    data = np.loadtxt(sim_filename, dtype=np.str, delimiter="\t")
    uids = set(data[:, 0].flatten())
    bids = set(data[:, 1].flatten())
    # uid2ind = {v: k for k, v in enumerate(uids)}
    uid2ind = {int(v): k for k, v in enumerate(uids)}
    ind2uid = reverse_map(uid2ind)
    # bid2ind = {v: k for k, v in enumerate(bids)}
    bid2ind = {int(v): k for k, v in enumerate(bids)}
    ind2bid = reverse_map(bid2ind)

    data[:, 0] = [uid2ind[int(r)] for r in data[:, 0]]
    data[:, 1] = [bid2ind[int(r)] for r in data[:, 1]]

    # data[:, 0] = [uid2ind[r] for r in data[:, 0]]
    # data[:, 1] = [bid2ind[r] for r in data[:, 1]]

    print('finish load data from %s, cost %.2f seconds, users: %s, items=%s' % (sim_filename, time.time() - start_time, len(uids), len(bids)))
    # must convert data type to float
    data = data.astype(dtype=np.float)
    print("data shape: ", data.shape, data.dtype)

    eps, lamb, iters = 10, 10, 500
    print('start generate mf features, (K, eps, reg, iters) = (%s, %s, %s, %s)' % (K, eps, lamb, iters))
    mf = MF(data=data, train_data=data, test_data=[], K=K, eps=eps, lamb=lamb, max_iter=iters, call_logger=logger)
    U,V = mf.run()

    start_time = time.time()
    wfilename = os.path.join(data_dir, 'mf_features/path_count/%s_user.dat' % (path_str))
    if use_topK:
        wfilename = os.path.join(data_dir, 'mf_features/path_count/%s_top%s_user.dat' % (path_str, topK))

    fw = open(wfilename, 'w+')
    res = []
    for ind, fs in enumerate(U):
        row = []
        row.append(ind2uid[ind])
        row.extend(fs.flatten())
        res.append('\t'.join([str(t) for t in row]))

    fw.write('\n'.join(res))
    fw.close()
    print('User-Features: %s saved in %s, cost %.2f seconds' % (U.shape, wfilename, time.time() - start_time))

    start_time = time.time()
    wfilename = os.path.join(data_dir, 'mf_features/path_count/%s_item.dat' % (path_str))
    if use_topK:
        wfilename = os.path.join(data_dir, 'mf_features/path_count/%s_top%s_item.dat' % (path_str, topK))

    fw = open(wfilename, 'w+')
    res = []
    for ind, fs in enumerate(V):
        row = []
        row.append(ind2bid[ind])
        row.extend(fs.flatten())
        res.append('\t'.join([str(t) for t in row]))

    fw.write('\n'.join(res))
    fw.close()
    print('Item-Features: %s  saved in %s, cost %.2f seconds' % (V.shape, wfilename, time.time() - start_time))


def run_movie():
    print("run start...")

    for path_str in ['UPBPersonB', 'UPBTypeB']:
        run(path_str)
        # break

    for path_str in ['UPBUB', 'UNBUB']:
        run(path_str)

    for path_str in ['ratings_only']:
        run(path_str)

    print("run over")


if __name__ == '__main__':
    log_filename = 'log/%s_mf_feature_geneartion_%s.log' % ("movie", "all")
    exp_id = int(time.time())
    logger = init_logger('exp_%s' % exp_id, log_filename, logging.INFO, False)
    print('data: %s, path_str: %s' % (data_dir, "all"))
    logger.info('data: %s, path_str: %s', data_dir, "all")
    run_movie()
