import sys
import os
import pickle


def make_write_safe(text):
    return text.encode('latin-1', 'ignore').decode('utf-8', 'ignore')


def write_to_file(text, filename, end='\n'):
    with open(filename, mode='w') as file:
        file.write(make_write_safe(text) + end)


if __name__ == '__main__':
    data_filename = sys.argv[1]
    lead_3_output_dir = sys.argv[2]
    byte_limit = None
    if not os.path.exists(lead_3_output_dir):
        os.makedirs(lead_3_output_dir)
    with open(data_filename, mode='rb') as file:
        data = pickle.load(file)

    count = 0
    for doc in data:
        count += 1
        pred_filename = '{}/{:0>6}_hypothesis.txt'.format(lead_3_output_dir, count)
        pred_file = open(pred_filename, 'w')
        text = ''
        for sent, label in doc[0][:3]:
            text += make_write_safe(sent.strip() + '\n')
        if byte_limit:
            text = text[:byte_limit]
        print(text + '\n', file=pred_file)
        file.close()
