"""Transform a file in an action sequence.

Usage:
    compute_action_sequence [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    --max-num-files INT  number of files to load.
    -f --is_file         the path is a file (and not a folder)
    --debug              debug mode [default: False]
"""
from docopt import docopt

import os, sys
# Add parent directory dynamically
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset import load_data_file, get_data_files_from_directory

if __name__ == '__main__':
    args = docopt(__doc__)

    # hte path is relative to where you called the file from
    with open('./test_outputs/action_sequence.txt', 'w') as output:
        data_files = get_data_files_from_directory(
            [args["CORPUS_DATA_DIR"]],
            max_num_files = args.get("--max-num-files")
        )
        for i, file_path in enumerate(data_files):
            print(f"{i}:  Processing {file_path}")
            action_lists, node_lists, fathers_lists = load_data_file(file_path, as_string=True)

            output.write(f"----{file_path}\n")
            for i in range(len(action_lists)):
                for j in range(len(action_lists[i])):
                    output.write("Pos:%4d | Father:%4d | %-20s  |  %-30s \n" % (j, fathers_lists[i][j], node_lists[i][j], action_lists[i][j]))

                output.write("\n")