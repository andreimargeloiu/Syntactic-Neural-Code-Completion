"""Extract and save grammar
Usage:
    test_compute_grammar.py [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    -v --verbose         show unnecessary extra information.
    -f --is_file         the path is a file (and not a folder)
    --debug              debug mode [default: False]
"""

from docopt import docopt

from dataset import build_grammar_from_data_dir
from grammar import Grammar

if __name__ == '__main__':
    args = docopt(__doc__)

    grammar = build_grammar_from_data_dir(args['CORPUS_DATA_DIR'])

    grammar.save()
    grammar_loaded = Grammar.load()
    assert str(grammar) == str(grammar_loaded)
    print("Grammar successfully saved.")
