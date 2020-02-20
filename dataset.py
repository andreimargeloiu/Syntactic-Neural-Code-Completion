import os
from glob import iglob
from typing import List, Optional, Iterable
from collections import Counter

from docopt import docopt
from dpu_utils.mlutils import Vocabulary

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


def load_data_file(file_path: str, as_string=False) -> Iterable[List[str]]:
    """
        Load a single data file, returning a stream of rules, corresponding to the action sequence.

        Args:
            file_path: The path to a data file.

        Returns:
            Iterable of lists of strings, each a list of tokens observed in the data.
    """
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())

        return TreeNode.from_graph(g).to_action_sequence(as_string=as_string)


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
        rules = load_data_file(file_path, as_string=True)
        for rule in rules:
            counter[rule] += 1

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
    def load_data_from_dir(
            vocab: Vocabulary, length: int, data_dir: str, max_num_files: Optional[int] = None
    ) -> np.ndarray:
        """
        Load and tensorise data

        Returns:
            numpy int32 array of shape [None, length], containing the tensorised data
        """

        # TODO


if __name__ == '__main__':
    args = docopt(__doc__)

    # create grammar

    # create vocab

    # create training set
