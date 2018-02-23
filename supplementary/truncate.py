import sys


def write_truncated(src_filename, dest_filename, num_bytes):
    with open(src_filename, mode='r') as file:
        truncated_lines = [line.strip()[:num_bytes] + '\n' for line in file]
    with open(dest_filename, mode='w') as file:
        file.writelines(truncated_lines)


if __name__ == '__main__':
    src_filename = sys.argv[1]
    dest_filename = sys.argv[2]
    num_bytes = int(sys.argv[3])
    write_truncated(src_filename, dest_filename, num_bytes=num_bytes)
