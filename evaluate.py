#!/usr/bin/env python
"""
Usage:
    evaluate.py [options]

Options:
    -h --help                        Show this screen.
    --model=NAME                     The model version
    --saved-data-dir=NAME            The path to the saved data.
    --debug                          Enable debug routines. [default: False]
    --trained-model=NAME             Path to trained model
"""
import os
import pickle

from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import load_data_from_dir, get_minibatch_iterator
from model import SyntacticModelv2, SyntacticModelv1, SyntacticModelv3


def evaluate(args) -> list:
    print("Loading model ...")
    if args['--model'] == 'v1':
        model = SyntacticModelv1.restore(args["--trained-model"])
    elif args['--model'] == "v2":
        model = SyntacticModelv2.restore(args["--trained-model"])
    elif args['--model'] == "v3":
        model = SyntacticModelv3.restore(args["--trained-model"])

    print(f"  Loaded trained model from {args['--trained-model']}.")

    print("Loading data ...")
    # with open(os.path.join(args['--saved-data-dir'], 'train_data'), 'rb') as input:
    #     train_data = pickle.load(input)
    # print(f"  Loaded {train_data[0].shape[0]} training samples.")
    with open(os.path.join(args['--saved-data-dir'], 'valid_data'), 'rb') as input:
        valid_data = pickle.load(input)
    print(f"  Loaded {valid_data[0].shape[0]} validation samples.")
    # with open(os.path.join(args['--saved-data-dir'], 'seen_test_data'), 'rb') as input:
    #     seen_test_data = pickle.load(input)
    # print(f"  Loaded {seen_test_data[0].shape[0]} seen test samples.")
    # with open(os.path.join(args['--saved-data-dir'], 'unseen_test_data'), 'rb') as input:
    #     unseen_test_data = pickle.load(input)
    # print(f"  Loaded {unseen_test_data[0].shape[0]} unseen test samples.")


    accs = []
    for dataset, name in zip([valid_data], ['valid_data']):
    # for dataset, name in zip([train_data, valid_data, seen_test_data, unseen_test_data], ['train_data', 'valid_data', 'seen_test_data', 'unseen_test_data']):
        test_loss, test_acc = model.run_one_epoch(
            get_minibatch_iterator(
                dataset,
                model.hyperparameters["batch_size"],
                is_training=False,
                drop_remainder=False,
            ),
            training=False,
        )
        print(f"{name}:  Loss {test_loss:.4f}, Acc {test_acc:.3f}")
        accs.append(test_acc)

    return accs

if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: evaluate(args), args["--debug"])
