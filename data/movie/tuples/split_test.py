# -*- coding:utf-8 -*-
import codecs
import sys

if __name__ == "__main__":
    file_reader = codecs.open(sys.argv[1], mode="r", encoding="utf-8")
    pair_nums = 299411
    split_num = int(pair_nums/5)
    line = ""
    line = file_reader.readline()
    for i in range(4):
        file_writer = codecs.open(sys.argv[2] + "_part_" + str(i) + ".txt", mode="w", encoding="utf-8")
        temp_num = 0
        while line:
            file_writer.write(line)
            line = file_reader.readline()
            temp_num += 1
            if temp_num > split_num*101:
                break
        file_writer.close()

    file_writer = codecs.open(sys.argv[2] + "_part_" + str(4) + ".txt", mode="w", encoding="utf-8")
    while line:
        file_writer.write(line)
        line = file_reader.readline()
    file_reader.close()
