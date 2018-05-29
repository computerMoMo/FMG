# -*- coding:utf-8 -*-
'''
    calculate the similarity by commuting matrix baesd on meta-structure
'''
from __future__ import print_function
from numba import jit
from scipy.sparse import csr_matrix as csr
from utils import reverse_map, generate_adj_mat, save_triplets
from cal_commuting_mat import *

import time
import sys
import os
import cPickle as pickle
import numpy as np
import bottleneck as bn


data_dir = "data/movie"


def get_topK_items(comm_res, ind2uid, ind2bid, topK=1000):
    start = time.time()
    U, _ = comm_res.shape
    triplets = []
    for i in xrange(U):
        items = comm_res.getrow(i).toarray().flatten()
        cols = np.argpartition(-items, topK).flatten()[:topK]
        cols = [c for c in cols if items[c] > 0]
        triplets.extend([(ind2uid[i], ind2bid[c], items[c]) for c in cols])
    print('get top %s items, total %s entries, cost %.2f seconds' % (topK, len(triplets), time.time() - start))
    return triplets


def get_bo(path_str, bid2ind):
    sfilename = ""
    if 'Type' in path_str:
        sfilename = os.path.join(data_dir, "tuples/movie_belong_type.txt")
    elif "Person" in path_str:
        sfilename = os.path.join(data_dir, "tuples/movie_category_person.txt")

    lines = open(sfilename, 'r').readlines()
    parts = [l.strip().split("\t") for l in lines]
    bos = [(b, o) for b, o in parts]
    ond2ind = {v: k for k, v in enumerate(set([o for _, o in bos]))}
    ind2ond = reverse_map(ond2ind)
    adj_bo, adj_bo_t = generate_adj_mat(bos, bid2ind, ond2ind)
    return adj_bo, adj_bo_t


def load_eids(eid_filename):
    lines = open(eid_filename, 'r').readlines()
    eids = [l.strip() for l in lines]
    eid2ind = {v:k for k,v in enumerate(eids)}
    ind2eid = reverse_map(eid2ind)
    print ('get %s entities from %s' %(len(eids), eid_filename))
    return eids, eid2ind, ind2eid


'''
    calculate the commuting matrix in U-B-*-B style
'''
def cal_comm_mat_UBB(path_str):
    print("path str:", path_str)

    uid_filename = os.path.join(data_dir, "entity_ids/user_id.txt")#users
    print('run cal_comm_mat_samples for users in ', uid_filename)
    lines = open(uid_filename, 'r').readlines()
    uids = [l.strip() for l in lines]
    uid2ind = {v: k for k, v in enumerate(uids)}
    ind2uid = reverse_map(uid2ind)

    bid_filename = os.path.join(data_dir, "entity_ids/movie_id.txt")#items
    lines = open(bid_filename, 'r').readlines()
    bids = [l.strip() for l in lines]
    bid2ind = {v: k for k, v in enumerate(bids)}
    ind2bid = reverse_map(bid2ind)

    upb_filename = os.path.join(data_dir, "tuples/user_rate_movie.txt")#positive rating
    upb = np.loadtxt(upb_filename, dtype=str, delimiter="\t")

    # generate users items adjacency matrix
    adj_ub, adj_ub_t = generate_adj_mat(upb, uid2ind, bid2ind)

    print (adj_ub.shape, adj_ub_t.shape)

    # generate items object adjacency matrix
    adj_bo, adj_bo_t = get_bo(path_str, bid2ind)
    print (adj_bo.shape, adj_bo_t.shape)

    t1 = time.time()
    # compute u-> b -> o <- b
    comm_res = cal_mat_ubb(path_str, adj_ub, adj_bo, adj_bo_t)
    t2 = time.time()
    print ('cal res of %s cost %2.f seconds' % (path_str, t2 - t1))
    print ('comm_res shape=%s,densit=%s' % (comm_res.shape, comm_res.nnz * 1.0/comm_res.shape[0]/comm_res.shape[1]))

    K = 500
    wfilename = os.path.join(data_dir, 'sim_res/path_count/%s_top%s.res' % (path_str, K))
    triplets = get_topK_items(comm_res, ind2uid, ind2bid, topK=K)

    save_triplets(wfilename, triplets)
    t3 = time.time()
    print ('save res of %s cost %2.f seconds' % (path_str, t3 - t2))



'''
    calculate commuting matrix for U-*-U-pos-B style
'''
def cal_comm_mat_UUB(path_str):
    print("path str:", path_str)

    uid_filename = os.path.join(data_dir, "entity_ids/user_id.txt")
    bid_filename = os.path.join(data_dir, "entity_ids/movie_id.txt")
    upb_filename = os.path.join(data_dir, "tuples/user_rate_movie.txt")

    print('cal commut mat for %s, filenames: %s, %s, %s' % (path_str, uid_filename, bid_filename, upb_filename))

    uids, uid2ind, ind2uid = load_eids(uid_filename)
    bids, bid2ind, ind2bid = load_eids(bid_filename)

    upb = np.loadtxt(upb_filename, dtype=np.str, delimiter="\t")
    adj_upb, adj_upb_t = generate_adj_mat(upb, uid2ind, bid2ind)

    if path_str == "UPBUB":
        start = time.time()
        UBU = adj_upb.dot(adj_upb_t)
        print('UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0/UBU.shape[0]/UBU.shape[1], time.time() - start))

    elif path_str == "UNBUB":
        unb_filename = os.path.join(data_dir, "tuples/user_neg_movie.txt")
        unb = np.loadtxt(unb_filename, dtype=np.str)
        adj_unb, adj_unb_t = generate_adj_mat(unb, uid2ind, bid2ind)
        start = time.time()
        UBU = adj_unb.dot(adj_unb_t)
        print('UBU(%s), density=%.5f cost %.2f seconds' % (UBU.shape, UBU.nnz * 1.0 / UBU.shape[0] / UBU.shape[1], time.time() - start))

    else:
        raise Exception("path str "+path_str+" not support")

    start = time.time()
    UBUB = UBU.dot(adj_upb)
    print('UBUB(%s), density=%.5f cost %.2f seconds' % (UBUB.shape, UBUB.nnz * 1.0/UBUB.shape[0]/UBUB.shape[1], time.time() - start))

    start = time.time()
    K = 500
    triplets = get_topK_items(UBUB, ind2uid, ind2bid, topK=K)
    wfilename = os.path.join(data_dir, 'sim_res/path_count/%s_top%s.res' % (path_str, K))
    save_triplets(wfilename, triplets)
    print('finish saving %s %s entries in %s, cost %.2f seconds' % (len(triplets), path_str, wfilename, time.time() - start))


if __name__ == '__main__':
    print("start...\n")
    cal_comm_mat_UBB("UPBTypeB")
    print("\n")
    cal_comm_mat_UBB("UPBPersonB")
    print("\n")
    cal_comm_mat_UUB("UPBUB")
    print("\n")
    cal_comm_mat_UUB("UNBUB")
