"""Transform a file in an action sequence.

Usage:
    compute_action_sequence [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    -f --is_file         the path is a file (and not a folder)
    --debug              debug mode [default: False]
"""

from docopt import docopt

from dataset import load_data_file, get_data_files_from_directory

if __name__ == '__main__':
    args = docopt(__doc__)

    with open('test_outputs/action_sequence.txt', 'w') as output:
        data_files = get_data_files_from_directory(args["CORPUS_DATA_DIR"])
        for file_path in data_files:
            actions = load_data_file(file_path, as_string=True)

            output.write(f"----{file_path}\n")
            for action in actions:
                output.write(action)
                output.write('\n')
