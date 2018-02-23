import sys
from rouge import FilesRouge

if __name__ == '__main__':
    hypothesis_path = sys.argv[1]
    reference_path = sys.argv[2]
    files_rouge = FilesRouge(hypothesis_path, reference_path)
    out = files_rouge.get_scores(avg=True)
    print('ROUGE-1: ', out['rouge-1']['f'])
    print('ROUGE-2: ', out['rouge-2']['f'])
    print('ROUGE-L: ', out['rouge-l']['f'])
