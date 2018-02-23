import os


def dump_sentences(src_dir, des_filename):
    with open(des_filename, 'w', encoding='utf-8') as dest_file:
        for root, dirs, files in os.walk(src_dir):
            for fname in files:
                print(os.path.join(root, fname))
                try:
                    with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                        text = f.readlines()
                except UnicodeDecodeError:
                    print('Error while reading file --------->\t', os.path.join(root, fname))
                    continue
                i = 2
                j = 1
                sentences = []
                while j <= 3:

                    while i < len(text) and text[i] != '\n':
                        if j == 1:
                            words = text[i].replace('@', '').split()
                            st = " ".join(words[:-1])
                            st.replace('\t', ';')
                            sentences.append(st + '.\n')
                        if j == 2:
                            lords = text[i].replace('@', '').split()
                            lords = " ".join(lords)
                            lords.replace('\n', ' ')
                            sentences.append(lords + '.\n')
                        i += 1
                    i = i + 1
                    j += 1
                dest_file.writelines(sentences)


src_dir = '/media/pranav/Data/Downloads/Datasets/Summarization/neuralsum'
des_dir = 'data/dump/sentences.txt'
dump_sentences(src_dir, des_dir)
