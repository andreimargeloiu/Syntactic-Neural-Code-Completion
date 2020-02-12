"""Main function to obtain to conpute the grammar
Usage:
    build_grammar.py [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    -v --verbose         show unnecessary extra information.
    -f --file            the path is a file (and not a folder)
    --debug              debug mode [default: False]
"""


from docopt import docopt

from proto.graph_pb2 import Graph


def hello_world():
    print("Hello World")
    print("-----")

# TODO
def compute_grammar():
    return None

def get_graph_from_file(file_path):
    with open(file_path, 'r') as f:
        g = Graph()
        g.ParseFromString(f.read())

        print(g)

if __name__ == "__main__":
    args = docopt(__doc__)
    hello_world()
    print(args)