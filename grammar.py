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
import math
import os
import re

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
        self.children = tuple(children)  # use tuple instead list because it allows to use __hash__()

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
        # all nodes should have a startPosition and endPosition
        assert feature_node.startPosition != -1 and feature_node.endPosition != -1

        self.feature_node = feature_node  # FeatureNode extracted from the .proto file
        self.children = children  # List of children sorted from left to right

    @property
    def contents(self):
        return self.feature_node.contents

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_leaf

    def get_node_rule(self):
        """
        Return the rule of this node
        """
        # If node is leaf or preterminal, t
        if self.is_leaf or len(self.children) == 0:
            return []
        else:
            # compute the
            children_contents = []
            for child in self.children:
                # IDENTIFIER_TOKEN have the .contents as the token name "test_foo"
                # However, I will add IDENTIFIER_TOKEN instead of the identifier name,
                # because we'll not predict the identifier's name

                # For IdentifierTokens add IDENTIFIER_TOKEN
                if child.feature_node.type == FeatureNode.NodeType.IDENTIFIER_TOKEN:
                    children_contents.append(token_names[child.feature_node.type])
                # For Tokens, the compiler doesn't give very precise outputs
                # Clear signs such as PLUS, EQ are in capital letters.
                # Constants (e.g., 102, 'my_string') have the type TOKEN, while
                # also things from the language 'int', 'void' are of type token
                # However, as I can't distinguish between them, I will consider
                # everything that is capital case to be from the language (thus
                # store their .contents) and everything which is not capital, I
                # will store the type TOKEN
                elif child.feature_node.type == FeatureNode.NodeType.TOKEN:
                    if all(x.isupper() for x in child.feature_node.contents): # If it's a Java token eg: PLUS, EQ
                        children_contents.append(child.contents)
                    else:
                        children_contents.append(token_names[child.feature_node.type])
                else:
                    children_contents.append(child.contents)

            return [Rule(self.contents, children_contents)]

    def compute_rules(self) -> [Rule]:
        """
        :return: list of rules from this subtree
        """
        # Rule of this node
        rules = self.get_node_rule()

        # Rules of children
        for child in self.children:
            rules.extend(child.compute_rules())

        return rules

    def to_action_sequence(self):
        """
        Decompose each method subtree into a sequence of actions.
        """
        methods_actions = []

        return methods_actions

    @staticmethod
    def from_graph(node_id: int, nodes_dict: dict, children_dict: dict):
        """
        Create this subtree.
        """
        node = nodes_dict[node_id]
        # print(f"node_id: {node_id} with children {children_dict[node_id]}")

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

    def add_rules(self, path):
        """
        Compute the rules from a file and add them to this Grammar
        """
        with open(path, 'rb') as f:
            g = Graph()
            g.ParseFromString(f.read())

            root = create_tree(g)
            for rule in root.compute_rules():
                grammar.rules.add(rule)

    @staticmethod
    def create_grammar(path, is_file=True):
        """
        Create grammar from some given .proto files
        """
        # TODO rules can have non-deterministic length on the RHS

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
                    print(rule)
                    grammar.rules.add(rule)

        return grammar

    def save(self, file_name='grammar_rules.txt'):
        with open(file_name, 'w') as output:
            output.write(self.__repr__())

    @staticmethod
    def load(file_name='grammar_rules.txt'):
        grammar = Grammar()
        with open(file_name, 'r') as f:
            for line in f:
                rules_split = re.compile("->|,").split(line)
                rules_split = [str.strip(str(elem), ' \n') for elem in rules_split]

                assert len(rules_split) > 1

                grammar.rules.add(Rule(rules_split[0], rules_split[1:]))

        return grammar

    def __repr__(self):
        # sort rules string
        rules_strings = []
        for rule in list(self.rules):
            rules_strings.append(str(rule))
        rules_strings.sort()

        return '\n'.join(str(rule) for rule in rules_strings)


# Auxiliary methods
def create_tree(g: Graph):
    """
    Converts a Graph into a TreeNode
    """
    # create dictionary with notes and edges
    nodes_dict = dict()
    edges_dict = dict()
    children_dict = dict()
    for node in g.node:
        nodes_dict[node.id] = node
        edges_dict[node.id] = []
        children_dict[node.id] = []

    for edge in g.edge:
        edges_dict[edge.sourceId].append(edge)  # edges for the intermediate graph to fill the startPosition
        if edge.type in (FeatureEdge.EdgeType.AST_CHILD, FeatureEdge.EdgeType.ASSOCIATED_TOKEN):  # edges for the TreeNode
            children_dict[edge.sourceId].append(edge.destinationId)

    # fill the missing startPosition and endPosition
    for node in g.node:
        modify_startend_positions(node.id, nodes_dict, edges_dict)

    # sort the list of children based on the starting position.
    for node in g.node:
        children_dict[node.id].sort(key=lambda x: (nodes_dict[x].startPosition, nodes_dict[x].endPosition))

    return TreeNode.from_graph(0, nodes_dict, children_dict)


def modify_startend_positions(node_id, nodes_dict, edges_dict):
    """
    Recursively iterates the tree and updates the startPosition
    and endPosition with that of the subtree.
    """
    node = nodes_dict[node_id]
    if node.startPosition != -1:
        return node.startPosition, node.endPosition

    start_position = (int)(1e8)
    end_position = -1

    for edge in edges_dict[node_id]:
        child_start, child_end = modify_startend_positions(edge.destinationId, nodes_dict, edges_dict)
        start_position = min(start_position, child_start)
        end_position = max(end_position, child_end)

    # modify this node
    nodes_dict[node_id].startPosition = start_position
    nodes_dict[node_id].endPosition = end_position

    return start_position, end_position


def is_token(node: FeatureNode):
    return node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN)


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

    # Assert saving and loading a grammar works
    grammar = Grammar.create_grammar(args['CORPUS_DATA_DIR'], args['--is_file'])
    grammar.save()
    grammar_loaded = Grammar.load()
    assert str(grammar) == str(grammar_loaded)
    print("Grammar successfully saved.")
