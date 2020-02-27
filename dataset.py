import os
from glob import iglob
from typing import List, Optional, Iterable, Iterator, Tuple
from collections import Counter

import numpy as np
from docopt import docopt
from dpu_utils.mlutils import Vocabulary
from more_itertools.more import chunked

from grammar import Grammar, TreeNode
from graph_pb2 import Graph

DATA_FILE_EXTENSION = "proto"
START_SYMBOL = "%START%"
END_SYMBOL = "%END%"


def get_data_files_from_directory(data_dir: str, max_num_files: Optional[int] = None) -> List[str]:
    """
    Get list of paths for file .proto of Graph
    """
    # get a generator for all files matching the extensions
    files = iglob(
        os.path.join(data_dir, "**/*.%s" % DATA_FILE_EXTENSION), recursive=True
    )
    if max_num_files:
        files = sorted(files)[: int(max_num_files)]
    else:
        files = list(files)
    return files


def get_methods_action_sequences(node: TreeNode):
    """
    Return list of nodes from subtree which are of type Method.
    """
    if node.contents == "METHOD":
        return [node]

    result = []
    for child in node.children:
        result.extend(get_methods_action_sequences(child))

    return result


def load_data_file(file_path: str, as_string=True) -> (Iterable[List[str]], Iterable[List[str]]):
    """
        Returning a lists of sequences of rules corresponding to the METHODS in a file.
        (thus don't consider tokens outside of the methods)

        Args:
            file_path: The path to a data file.

        Returns tupel of:
            1. Iterable of lists of strings, each list containing the ACTION RULES for the subtrees.
            2. Iterable of lists of strings, each list containing the NODES expanded for the subtrees.
    """
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())

        root = TreeNode.from_graph(g)
        method_nodes = get_methods_action_sequences(root)

        actions_list, nodes_list = [], []
        for node in method_nodes:
            actions, nodes = node.to_action_sequence_and_nodes(as_string=as_string)
            actions_list.append(actions)
            nodes_list.append(nodes)

        return actions_list, nodes_list


def build_vocab_from_data_dir(data_dir: str, vocab_size: int, max_num_files: Optional[int]) \
        -> Tuple[Vocabulary, Vocabulary]:
    """
    Build an Action Vocabulary and a Node Vocabulary
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)

    # Create vocabulary with START_SYMBOL and END_SYMBOL
    vocab_nodes = Vocabulary(add_unk=True, add_pad=True)
    vocab_nodes.add_or_get_id(START_SYMBOL)
    vocab_nodes.add_or_get_id(END_SYMBOL)

    # TODO the actions vocabulary should NOT contain unk and padding
    vocab_actions = Vocabulary(add_unk=True, add_pad=True)
    vocab_actions.add_or_get_id(START_SYMBOL)
    vocab_actions.add_or_get_id(END_SYMBOL)

    # Count actions and nodes
    counter_nodes, counter_actions = Counter(), Counter()
    for file_path in data_files:  # for each file, count all tokens
        action_lists, node_lists = load_data_file(file_path, as_string=True)

        for action_sequence, node_sequence in zip(action_lists, node_lists):
            for action, node in zip(action_sequence, node_sequence):
                counter_nodes[node] += 1
                counter_actions[action] += 1

    # Add the most common rules in the vocabulary
    for node, _ in counter_nodes.most_common(vocab_size - 2):
        vocab_nodes.add_or_get_id(node)

    for action, _ in counter_actions.most_common(vocab_size):
        vocab_actions.add_or_get_id(action)

    return vocab_nodes, vocab_actions


def build_grammar_from_data_dir(data_dir: str, max_num_files: Optional[int] = None) -> Grammar:
    """
    Create Grammar
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)
    return Grammar.create_grammar(data_files)


def tensorise_token_sequence(
        vocab: Vocabulary, length: int, token_seq: Iterable[str],
) -> List[int]:
    """
    Tensorise a single example (Transform the token sequence into a sequence of IDs of fixed length)

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        token_seq: Sequence of tokens to tensorise.

    Returns:
        List with length elements that are integer IDs of tokens in our vocab.
    """
    token_ids = [vocab.get_id_or_unk(START_SYMBOL)]
    token_ids.extend(vocab.get_id_or_unk_multiple(token_seq, pad_to_size=length - 1))

    # END_SYMBOL must be the last element in the tokenised sequence
    end_position = min(1 + len(token_seq), length - 1)
    token_ids[end_position] = vocab.get_id_or_unk(END_SYMBOL)

    return token_ids


def load_data_from_dir(
        vocab: Vocabulary, length: int, data_dir: str, max_num_files: Optional[int] = None
) -> np.ndarray:
    """
    Load and tensorise data.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        data_dir: Directory from which to load the data.
        max_num_files: Number of files to load at most.

    Returns:
        numpy int32 array of shape [None, length], containing the tensorised
        data.
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)

    tensorised_result = []
    for data_file in data_files:
        actions_seq, nodes_seq = load_data_file(data_file)

        for action_seq, node_seq in zip(actions_seq, nodes_seq):
            tensorised_result.append(tensorise_token_sequence(vocab, length, action_seq))

    return np.array(tensorised_result, dtype=np.int32)


def get_minibatch_iterator(
    token_seqs: np.ndarray,
    batch_size: int,
    is_training: bool,
    drop_remainder: bool = True
) -> Iterator[np.ndarray]:
    """
    Create an iterator for a minibatch by shuffling the token sequences.
    """
    indices = np.arange(token_seqs.shape[0])
    if is_training:
        np.random.shuffle(indices)

    for minibatch_indices in chunked(indices, batch_size):
        if len(minibatch_indices) < batch_size and drop_remainder:
            break # Drop last, smaller batch

        minibatch_seqs = token_seqs[minibatch_indices]
        yield minibatch_seqs