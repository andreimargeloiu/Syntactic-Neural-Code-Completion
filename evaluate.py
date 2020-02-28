#!/usr/bin/env python
"""
Usage:
    evaluate.py [options]

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
    --trained-model=NAME             Path to trained model
    --test-dir=NAME                  Path to training directory
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import load_data_from_dir, get_minibatch_iterator
from model import SyntacticModel


def run(arguments) -> None:
    print("Loading data ...")
    model = SyntacticModel.restore(arguments["--trained-model"])
    print(f"  Loaded trained model from {arguments['--trained-model']}.")

    test_data_nodes, test_data_actions = load_data_from_dir(
        model.vocab_nodes,
        model.vocab_actions,
        length=model.hyperparameters["max_seq_length"],
        data_dir=arguments["--test-dir"],
        max_num_files=arguments.get("--max-num-files"),
    )
    print(
        f"  Loaded {test_data_actions.shape[0]} test samples from {arguments['--test-dir']}."
    )

    test_loss, test_acc = model.run_one_epoch(
        get_minibatch_iterator(
            test_data_actions,
            model.hyperparameters["batch_size"],
            is_training=False,
            drop_remainder=False,
        ),
        training=False,
    )
    print(f"Test:  Loss {test_loss:.4f}, Acc {test_acc:.3f}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
