import os
from glob import iglob
from typing import List, Optional, Iterable, Iterator
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


def load_data_file(file_path: str, as_string=True) -> Iterable[List[str]]:
    """
        Load a single data file, returning a stream of rules
        corresponding to the action sequence for METHODS.
        (thus don't consider tokens outside of the methods)

        Args:
            file_path: The path to a data file.

        Returns:
            Iterable of lists of strings, each a list of tokens observed in the data.
    """
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())

        root = TreeNode.from_graph(g)
        method_nodes = get_methods_action_sequences(root)

        result = []
        for node in method_nodes:
            result.append(node.to_action_sequence(as_string=as_string))

        return result


def build_vocab_from_data_dir(data_dir: str, vocab_size: int, max_num_files: Optional[int]) -> Vocabulary:
    """
    Build the Vocabulary for a dataset
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)

    # Create vocabulary with START_SYMBOL and END_SYMBOL
    vocab = Vocabulary(add_unk=True, add_pad=True)
    vocab.add_or_get_id(START_SYMBOL)
    vocab.add_or_get_id(END_SYMBOL)

    # Compute Action sequences and add them to Vocabulary
    counter = Counter()
    for file_path in data_files:  # for each file, count all tokens
        action_sequences = load_data_file(file_path, as_string=True)

        for action_sequence in action_sequences:
            for action in action_sequence:
                counter[action] += 1

    # Add the most common rules in the vocabulary
    for elem, cnt in counter.most_common(vocab_size - 2):
        vocab.add_or_get_id(elem)

    return vocab


def build_grammar_from_data_dir(data_dir: str) -> Grammar:
    """
    Create Grammar
    """
    data_files = get_data_files_from_directory(data_dir)
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
    data = np.array(
        list(
            tensorise_token_sequence(vocab, length, token_seq)
            for data_file in data_files
            for token_seq in load_data_file(data_file)
        ),
        dtype=np.int32
    )
    return data


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