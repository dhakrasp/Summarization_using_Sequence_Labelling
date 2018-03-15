import sys
import os
import re
import pickle
from collections import OrderedDict


def get_entity_map(section):
    entity_map = {}
    for line in section.split('\n'):
        k = None
        try:
            k, v = line.strip().split(':', maxsplit=1)
        except ValueError:
            print('Line did not split...')
        if k is not None:
            entity_map[k] = v
    return OrderedDict(sorted(entity_map.items(), key=lambda x: x[0], reverse=True))


def extractive_summary_data(article_section):
    extractive_summary_data = []
    for line in article_section.split('\n'):
        sentence, label = re.split('[\t]+', line.strip())
        if sentence and label:
            label = int(label)
            extractive_summary_data.append([sentence, int(label)])
    return extractive_summary_data


def get_unanonymised_sentences_with_labels(dirname, anonymized=False):
    data = []
    count = 0
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            filename = os.path.join(root, fname)
            count += 1
            try:
                with open(filename, 'r') as f:
                    text = f.read()
                    if count % 1000 == 0:
                        print(count)
            except UnicodeDecodeError:
                print('Error while reading file --------->\t', fname)
                continue
            sections = text.split('\n\n')
            if not anonymized:
                enetity_map = get_entity_map(sections[-1])
                article_section = sections[1]
                for k, v in enetity_map.items():
                    article_section = article_section.replace(k, v)
                summary_section = sections[2]
                for k, v in enetity_map.items():
                    summary_section = summary_section.replace(k, v)
            doc = [extractive_summary_data(article_section),
                   summary_section.split('\n')]
            data.append(doc)
    return data


if __name__ == '__main__':
    dir_name = sys.argv[1]
    pickle_file = sys.argv[2]
    anonymized = False

    data = get_unanonymised_sentences_with_labels(dir_name, anonymized)
    # import pprint
    # pprint.pprint(data[0])
    # print(len(data))
    with open(pickle_file, mode='wb') as file:
        pickle.dump(data, file)
