import math
import os
import re
from collections import Iterable
from typing import List

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
        return str(self) == str(other)

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
        assert feature_node.startPosition != -1 or feature_node.endPosition != -1
        if feature_node.startPosition == -1 or feature_node.endPosition == -1:
            print(feature_node)

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
        if self.is_leaf:
            return None
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
                # Constants (e.g., 102, 'my_string') have type TOKEN, while
                # also things from the language 'int', 'void' are of type token
                # However, as I can't distinguish between them, I will consider
                # everything that is capital case to be from the language (and
                # store their .contents + ""_TOKEN) and everything which is not capital, I
                # will store the type TOKEN
                elif child.feature_node.type == FeatureNode.NodeType.TOKEN:
                    if all(x.isupper() for x in child.feature_node.contents):  # If it's a Java token eg: PLUS, EQ
                        children_contents.append(child.contents + "_TOKEN")
                    else:
                        children_contents.append(token_names[child.feature_node.type])
                else:
                    children_contents.append(child.contents)

            return Rule(self.contents, children_contents)

    def to_action_sequence(self, as_string=False) -> List[Rule]:
        """
        Decompose each method subtree into a sequence of actions.
        """
        if self.is_leaf:
            return []

        actions = [self.get_node_rule()]

        # Expansion of rules of children
        for child in self.children:
            actions.extend(child.to_action_sequence())

        if as_string:
            actions = list(map(str, actions))

        return actions

    @staticmethod
    def from_graph(g: Graph):
        """
        Convert a Graph into a TreeNode.

        Augment the Graph with startPosition/endPosition.
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
            if edge.type in (
                    FeatureEdge.EdgeType.AST_CHILD, FeatureEdge.EdgeType.RETURNS_TO):  # edges for the TreeNode
                children_dict[edge.sourceId].append(edge.destinationId)

        # fill the missing startPosition and endPosition
        for node in g.node:
            modify_startend_positions(node.id, nodes_dict, edges_dict)

        # sort the list of children based on the starting position.
        for node in g.node:
            children_dict[node.id].sort(key=lambda x: (nodes_dict[x].startPosition, nodes_dict[x].endPosition))

        return TreeNode.from_graph_dictionary(0, nodes_dict, children_dict)

    @staticmethod
    def from_graph_dictionary(node_id: int, nodes_dict: dict, children_dict: dict):
        """
        Convert the augmented dictionaries of a Graph into a TreeNode
        """
        node = nodes_dict[node_id]

        # corner case: leaf (token)
        if len(children_dict[node_id]) == 0:
            return TreeNode(node, [])

        # get TreeNodes for children
        children = []
        for child_id in children_dict[node_id]:
            children.append(TreeNode.from_graph_dictionary(child_id, nodes_dict, children_dict))

        return TreeNode(node, children)


class Grammar:
    """
    Class maintaining the grammar of the language, as extracted from the files
    """

    def __init__(self):
        self.rules = set()

    @staticmethod
    def create_grammar(file_paths):
        """
        Create grammar from folder

        :param: file_paths = paths to all .proto file to compute grammar
        """
        grammar = Grammar()
        for path in file_paths:
            with open(path, 'rb') as f:
                g = Graph()
                g.ParseFromString(f.read())

                root = TreeNode.from_graph(g)
                for rule in root.to_action_sequence():
                    grammar.rules.add(rule)

        return grammar

    def save(self, file_name='./test_outputs/grammar_rules.txt'):
        with open(file_name, 'w') as output:
            output.write(self.__repr__())

    @staticmethod
    def load(file_name='./test_outputs/grammar_rules.txt'):
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


def modify_startend_positions(node_id, nodes_dict, edges_dict):
    """
    Recursively iterates the tree and updates the startPosition
    and endPosition with that of the subtree.
    """
    node = nodes_dict[node_id]

    if node.startPosition != -1 and node.endPosition != -1:
        return node.startPosition, node.endPosition

    start_position = int(1e8)
    end_position = -1

    for edge in edges_dict[node_id]:
        if edge.destinationId <= node_id: # prevent infinite loop, because the AST children have higher ID
            continue

        child_start, child_end = modify_startend_positions(edge.destinationId, nodes_dict, edges_dict)
        start_position = min(start_position, child_start)
        end_position = max(end_position, child_end)

    # modify this node
    nodes_dict[node_id].startPosition = start_position
    nodes_dict[node_id].endPosition = end_position

    return start_position, end_position
