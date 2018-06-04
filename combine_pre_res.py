# -*- coding:utf-8 -*-
import codecs

if __name__ == "__main__":
    file_writer = codecs.open("fm_res/predict_res_total.txt", mode="w", encoding="utf-8")
    for i in range(5):
        file_reader = codecs.open("fm_res/predict_res_part_"+str(i)+".txt", mode="r")
        head_line = file_reader.readline()
        line = file_reader.readline()
        while line:
            file_writer.write(line)
            line = file_reader.readline()
        file_reader.close()
    file_writer.close()
