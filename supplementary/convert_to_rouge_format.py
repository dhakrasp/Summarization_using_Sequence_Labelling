import sys
import os
import nltk

dir_name = sys.argv[1]

hyp_filename = dir_name + '/hypothesis.txt'
ref_filename = dir_name + '/reference.txt'

hyp = []
with open(hyp_filename, mode='r') as file:
    hyp = [line.strip() for line in file]

ref = []
with open(ref_filename, mode='r') as file:
    ref = [line.strip() for line in file]

hyp_dir_name = dir_name + '/hypothesis/'
ref_dir_name = dir_name + '/reference/'

if not os.path.exists(hyp_dir_name):
    os.makedirs(hyp_dir_name)
if not os.path.exists(ref_dir_name):
    os.makedirs(ref_dir_name)

count = 1
for h, r in zip(hyp, ref):
    h_sents = nltk.sent_tokenize(h)
    filename = '{}{}_decoded.txt'.format(hyp_dir_name, count)
    with open(filename, mode='w') as file:
        for sent in h_sents:
            file.write(sent + '\n')
    r_sents = nltk.sent_tokenize(r)
    filename = '{}{}_decoded.txt'.format(ref_dir_name, count)
    with open(filename, mode='w') as file:
        for sent in r_sents:
            file.write(sent + '\n')
    count += 1
