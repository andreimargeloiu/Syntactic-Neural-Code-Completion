"""Compute Vocabulary
Usage:
    test_compute_vocabulary.py [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    --max-num-files INT  maximum number of files to consider
    -v --verbose         show unnecessary extra information.
    -f --is_file         the path is a file (and not a folder)
    --debug              debug mode [default: False]
"""
from docopt import docopt

import os, sys
# Add parent directory dynamically
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset import build_vocab_from_data_dir

if __name__ == '__main__':
    args = docopt(__doc__)

    vocab_nodes, vocab_actions = build_vocab_from_data_dir(args["CORPUS_DATA_DIR"], 500, args["--max-num-files"])

    with open('./test_outputs/vocabulary.txt', 'w') as output:
        output.write("Nodes vocabulary has size %d\n" % len(vocab_nodes))
        output.write(" %s [...]\n\n" % (str(vocab_nodes)[:1000]))

        output.write("Actions vocabulary has size %d\n" % len(vocab_actions))
        output.write(" %s [...]\n\n" % (str(vocab_actions)[:1000]))

