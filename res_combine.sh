#!/usr/bin/env bash
test_file_array=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0")
for file_id in ${test_file_array[@]}
do
    echo "combine on "$file_id
    score_file_name="fm_res/"$file_id".res"
    origin_file_name="data/movie/tuples/sample_test_data/user_item_test_"$file_id".txt"
    output_file_name="fm_res/test_result_"$file_id".txt"
    python res_combine.py $score_file_name $origin_file_name $output_file_name
    echo "evaluate..."
    score_file_name="fm_res/score_"$file_id".txt"
    python eval.py $output_file_name $score_file_name
done