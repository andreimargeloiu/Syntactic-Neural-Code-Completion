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
    --qualitative
    --validation-only                Call to do evaluation on validation only
"""
import os
import pickle
from collections import Counter

from docopt import docopt
from dpu_utils.utils import run_and_debug

import tensorflow.compat.v2 as tf
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
    with open(os.path.join(args['--saved-data-dir'], 'vocab_actions'), 'rb') as input:
        vocab_actions = pickle.load(input)

    if not args['--qualitative']:


        accs = []

        if args['--validation-only']:
            with open(os.path.join(args['--saved-data-dir'], 'valid_data'), 'rb') as input:
                valid_data = pickle.load(input)
            print(f"  Loaded {valid_data[0].shape[0]} validation samples.")
            datasets = zip([valid_data], ['valid_data'])
        else:
            with open(os.path.join(args['--saved-data-dir'], 'train_data'), 'rb') as input:
                train_data = pickle.load(input)
            print(f"  Loaded {train_data[0].shape[0]} training samples.")
            with open(os.path.join(args['--saved-data-dir'], 'valid_data'), 'rb') as input:
                valid_data = pickle.load(input)
            print(f"  Loaded {valid_data[0].shape[0]} validation samples.")
            with open(os.path.join(args['--saved-data-dir'], 'seen_test_data'), 'rb') as input:
                seen_test_data = pickle.load(input)
            print(f"  Loaded {seen_test_data[0].shape[0]} seen test samples.")
            with open(os.path.join(args['--saved-data-dir'], 'unseen_test_data'), 'rb') as input:
                unseen_test_data = pickle.load(input)
            print(f"  Loaded {unseen_test_data[0].shape[0]} unseen test samples.")

            datasets = zip([train_data, valid_data, seen_test_data, unseen_test_data],
                           ['train_data', 'valid_data', 'seen_test_data', 'unseen_test_data'])

        for dataset, name in datasets:
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

    if args['--qualitative']:
        with open(os.path.join(args['--saved-data-dir'], 'valid_data'), 'rb') as input:
            valid_data = pickle.load(input)
        print(f"  Loaded {valid_data[0].shape[0]} validation samples.")

        valid_data_iterator = get_minibatch_iterator(
            valid_data,
            batch_size=10,
            is_training=False,
            drop_remainder=True
        )

        aux = next(valid_data_iterator)
        good_predictions, bad_predictions, logits, targets = model.compute_loss_and_acc(
            model.compute_logits(tf.stack([*aux], axis=2), training=False),
            target_token_seq=aux[1],
            qualitative_results=True)

        good_predictions_counter = Counter(good_predictions.numpy())
        bad_predictions_counter = Counter(bad_predictions.numpy())

        print(f"GOOD predictions of model {args['--model']}")
        for node_id, count in good_predictions_counter.most_common(15):
            print("%5d   |   %15s" % (count, vocab_actions.get_name_for_id(node_id)))

        print(f"\nBAD predictions of model {args['--model']}")
        for node_id, count in bad_predictions_counter.most_common(15):
            print("%5d   |   %15s" % (count, vocab_actions.get_name_for_id(node_id)))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: evaluate(args), args["--debug"])
