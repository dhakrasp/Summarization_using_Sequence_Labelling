train_path="data_ext_abs/non_anon_ext_abs_train.pkl"
dev_path="data_ext_abs/non_anon_ext_abs_valid.pkl"
test_path="data_ext_abs/non_anon_ext_abs_test.pkl"
expt_dir="./experiment_glove_6B_100d_layers_2"
glove_path="glove_vectors/glove.6B.100d.npy"
num_layers=2
resume="True"
mode="train"

python3 summ_model.py --train_path $train_path \
--dev_path $dev_path \
--test_path $test_path \
--expt_dir $expt_dir \
--num_layers $num_layers \
--mode $mode \
