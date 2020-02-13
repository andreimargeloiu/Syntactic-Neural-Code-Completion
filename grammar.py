"""Extract and save grammar
Usage:
    grammar.py [options] CORPUS_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help            show this message and exit.
    -v --verbose         show unnecessary extra information.
    -f --is_file         the path is a file (and not a folder)
    --node_id=<n>       id of parent node (for debug) [default: -1]
    --debug              debug mode [default: False]
"""
import os
from docopt import docopt
from graph_pb2 import Graph, FeatureEdge, FeatureNode

token_names = {
    1: 'TOKEN',
    2: 'AST_ELEMENT',
    3: 'COMMENT_LINE',
    4: 'COMMENT_BLOCK',
    5: 'COMMENT_JAVADOC',
    6: 'AST_ROOT',
    7: 'IDENTIFIER_TOKEN',
    8: 'FAKE_AST',
    9: 'SYMBOL',
    10: 'SYMBOL_TYP',
    11: 'SYMBOL_VAR',
    12: 'SYMBOL_MTH',
    13: 'TYPE',
    14: 'METHOD_SIGNATURE',
    15: 'AST_LEAF'
}

class Rule:
    """
    A rule object inspired from context-free grammars.
    """

    def __init__(self, parent: str, children: [str]):
        assert parent is not None and children is not None and len(children) > 0

        self.parent = parent
        self.children = tuple(children) # use tuple instead list because it allows to use __hash__()

    def __eq__(self, other):
        return self.parent == other.parent and self.children == other.children

    def __hash__(self):
        return self.parent.__hash__() ^ self.children.__hash__()

    def __repr__(self):
        return '%s -> %s' % (self.parent, ', '.join([str(c) for c in self.children]))


class TreeNode:
    """
    Node of the parsed AST, which extends the FeatureNode with additional functionality.
    """

    def __init__(self, feature_node: FeatureNode, children):
        self.feature_node = feature_node  # FeatureNode extracted from the .proto file
        self.children = children  # List of children from left to right

    @property
    def contents(self):
        return self.feature_node.contents

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_leaf

    def compute_rules(self) -> [Rule]:
        """
        :return: list of rules from this subtree
        """
        # Corner case if it's leaf or preterminal, we don't care about the rule
        if self.is_leaf or self.is_preterminal:
            return []

        # Rule of this node
        rules = [Rule(self.contents, [child.contents for child in self.children])]

        # Rules of children
        for child in self.children:
            rules.extend(child.compute_rules())

        return rules

    @staticmethod
    def from_graph(node_id: int, nodes_dict: dict, children_dict: dict):
        """
        Create this subtree.
        """
        node = nodes_dict[node_id]
        print(f"node_id: {node_id} with children {children_dict[node_id]}")

        # corner case: leaf (token)
        if len(children_dict[node_id]) == 0:
            return TreeNode(node, [])

        # get TreeNodes for children
        children = []
        for child_id in children_dict[node_id]:
            children.append(TreeNode.from_graph(child_id, nodes_dict, children_dict))

        return TreeNode(node, children)


class Grammar:
    """
    Class maintaining the grammar of the language, as extracted from the files
    """

    def __init__(self):
        self.rules = set()

    @staticmethod
    def create_grammar(path, is_file=True):
        """
        Create grammar from some given .proto files
        """
        if not os.path.exists(path):
            raise Exception("File to create grammar does not exist")

        grammar = Grammar()

        if is_file:
            assert path.endswith('.proto')

            with open(path, 'rb') as f:
                g = Graph()
                g.ParseFromString(f.read())

                root = create_tree(g)
                for rule in root.compute_rules():
                    grammar.rules.add(rule)

        return grammar

    def save(self, file_name='grammar_rules.txt'):
        with open(file_name, 'w') as output:
            output.write(self.__repr__())

    @staticmethod
    def load(file_name):
        # TODO finish
        with open(file_name, 'r') as input:
            for line in input.readline():
                print(line)

    def __repr__(self):
        return '\n'.join(str(rule) for rule in self.rules)

# Auxiliary methods
def create_tree(g: Graph):
    """
    Converts the Graph into a TreeNode
    """
    # initialise children list for all nodes
    nodes_dict = dict()
    children_dict = dict()
    for node in g.node:
        nodes_dict[node.id] = node
        children_dict[node.id] = []

    for edge in g.edge:
        if edge.type == FeatureEdge.EdgeType.AST_CHILD:
            children_dict[edge.sourceId].append(edge.destinationId)

    # sort the list of children from left to right (the parse hives nodes ids in ascending order)
    for node in g.node:
        children_dict[node.id].sort()

    return TreeNode.from_graph(0, nodes_dict, children_dict)


class Debug:
    @staticmethod
    def print_all_edge(path, node_id):
        """
        Print all edges starting from node `node_id`
        """
        with open(path, 'rb') as f:
            g = Graph()
            g.ParseFromString(f.read())

            for edge in g.edge:
                if edge.sourceId == int(node_id):
                    print(edge)


if __name__ == '__main__':
    args = docopt(__doc__)

    # Debug.print_all_edge(args['CORPUS_DATA_DIR'], args['--node_id'])
    Grammar.create_grammar(args['CORPUS_DATA_DIR'], args['--is_file']).save()
