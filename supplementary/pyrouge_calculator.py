import sys
from pyrouge import Rouge155


def evaluate_rouge(hypothesis_dir, reference_dir):
    r = Rouge155()
    r.system_dir = hypothesis_dir
    r.model_dir = reference_dir
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_filename_pattern = '#ID#_reference.txt'
    output = r.convert_and_evaluate()
    return output, r.output_to_dict(output)


if __name__ == '__main__':
    hypothesis_dir = sys.argv[1]
    reference_dir = sys.argv[2]
    output, _ = evaluate_rouge(hypothesis_dir, reference_dir)
    print(output)
