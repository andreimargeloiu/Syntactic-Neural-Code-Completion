"""Main function to obtain to conpute the grammar
Usage:
    build_grammar.py [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    -v --verbose         show unnecessary extra information.
    -r --recursive       recursively iterate through files in path
    --debug              debug mode [default: False]
"""


from docopt import docopt

def hello_world():
    print("Hello World")
    print("-----")

if __name__ == "__main__":
    args = docopt(__doc__)
    hello_world()
    print(args)