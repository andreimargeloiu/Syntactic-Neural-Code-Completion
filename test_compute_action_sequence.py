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
import sys

from docopt import docopt

from dataset import load_data_file, get_data_files_from_directory

if __name__ == '__main__':
    args = docopt(__doc__)

    with open('test_outputs/action_sequence.txt', 'w') as output:
        data_files = get_data_files_from_directory(
            args["CORPUS_DATA_DIR"],
            max_num_files = args.get("--max-num-files")
        )
        for i, file_path in enumerate(data_files):
            print(f"{i}:  Processing {file_path}")
            action_sequences = load_data_file(file_path, as_string=True)

            output.write(f"----{file_path}\n")
            for action_sequence in action_sequences:
                for action in action_sequence:
                    output.write(action)
                    output.write('\n')
                output.write("\n")