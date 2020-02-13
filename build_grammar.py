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
from graph_pb2 import Graph



def get_graph_from_file(file_path):
    if not file_path.endswith('.java.proto'):
        print("Give the path only to files that end in .java.proto")
        exit(-1)
    with open(file_path, 'rb') as f:
        g = Graph()
        g.ParseFromString(f.read())

        aux_set = set()
        for node in g.node:
            aux_set.add(node.contents)

        print(aux_set)

if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)

    get_graph_from_file(args['CORPUS_DATA_DIR'])