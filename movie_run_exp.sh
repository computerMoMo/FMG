#!/usr/bin/env bash
test_file_array=("0.2" "0.4" "0.6" "0.8" "1.0")
for file_id in ${test_file_array[@]}
do
    file_name="user_item_test_"$file_id".txt"
    echo "split "$file_name
    python data/movie/tuples/split_test.py "data/movie/tuples/sample_test_data/"$file_name \
    "data/movie/tuples/sample_test_data/"$file_id
    for i in {0..4}
    do
        test_file_name="tuples/sample_test_data/"$file_id"_part_"$i".txt"
        test_res_name="data/movie/tuples/test_result/"$file_id"_part_"$i".res"
        echo "test file "$test_file_name
        python movie_run_exp.py config/yelp-50k.yaml -reg 0.5 -test_file_path $test_file_name \
        -test_res_save_path $test_res_name
    done
    echo "combine result"
    total_result_name="fm_res/"$file_id".res"
    part_res_name=""
    for i in {0..4}
    do
        part_res_name=$part_res_name"data/movie/tuples/test_result/"$file_id"_part_"$i".res "
    done
    cat $part_res_name > $total_result_name
    break
done
