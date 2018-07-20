# -*- coding:utf-8
from __future__ import print_function
import numpy as np
import codecs
import sys

if __name__ == "__main__":
    res_path = sys.argv[1]
    res_array = np.loadtxt(res_path, dtype=np.float)

    print(res_array.shape)

    test_file_path = sys.argv[2]
    test_res_path = sys.argv[3]
    file_reader = codecs.open(test_file_path, mode="r", encoding="utf-8")
    file_writer = codecs.open(test_res_path, mode="w", encoding="utf-8")
    line = file_reader.readline()
    file_writer.write("user_id\titem_id\tground_truth_score\tpredict_score\n")
    ids = 0
    while line:
        file_writer.write(line.strip()+"\t%.5f\n" % res_array[ids])
        ids += 1
        line = file_reader.readline()
    file_reader.close()
    file_writer.close()
