#!/bin/bash
output_dir_1=$1
output_dir_2=$2
echo $output_dir_1
sh evaluate.sh $output_dir_1
echo $output_dir_2
sh evaluate.sh $output_dir_2