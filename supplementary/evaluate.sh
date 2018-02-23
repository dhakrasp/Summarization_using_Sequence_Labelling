#!/bin/bash
output_dir=$1
echo 'Calculating full Rouge-F1'
python3 calculate_rouge_scores.py $output_dir/hypothesis.txt $output_dir/reference.txt > $output_dir/full_rouge_f.out &&
cat $output_dir/full_rouge_f.out
echo 'Done!'