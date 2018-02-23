import sys
import os
import nltk

hyp_dir_name = sys.argv[1]
ref_dir_name = sys.argv[2]

if not os.path.exists('tmp/'):
    os.makedirs('tmp/')
hyp_filename = 'tmp/hypothesis.txt'
ref_filename = 'tmp/reference.txt'

hyp = []
ref = []
for root, dirs, files in os.walk(hyp_dir_name):
    for fname in files:
        filename = os.path.join(root, fname)
        # print(filename)
        with open(filename, mode='r') as file:
            hyp.append(' '.join([line.strip() for line in file]))
            # print(hyp[-1])
        filename = os.path.join(root.replace(
            hyp_dir_name, ref_dir_name), fname.replace('decoded', 'reference'))
        # print(filename)
        with open(filename, mode='r') as file:
            ref.append(' '.join([line.strip() for line in file]))
            # print('')
            # print('')
            # print('')
            # print(ref[-1])
            # exit()

with open(hyp_filename, mode='w') as file:
    for line in hyp:
        file.write(line + '\n')
with open(ref_filename, mode='w') as file:
    for line in ref:
        file.write(line + '\n')
