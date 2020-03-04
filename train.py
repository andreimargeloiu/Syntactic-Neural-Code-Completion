"""
Usage:
    train.py [options]

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help                       Show this screen.
    --max-num-epochs EPOCHS         The maximum number of epochs to run [default: 100]
    --patience NUM                  Number of epochs to wait for the model improvement before stopping (for early stopping) [default: 5]
    --max-num-files INT             Number of files to load.
    --tensorboard-logs-path=NAME    Path to tensorboard logs
    --log-file=NAME
    --save-dir=NAME                 Save the models path
    --compute-data                  Flag for computing the training data
    --model=NAME                    Model type: v1, v2, v3
    --train-data-dir=NAME           Training directory path
    --saved-data-dir=NAME           Location of already computed data directory.
    --hypers-override HYPERS        JSON dictionary overriding hyperparameter values.
    --run-name NAME                 Picks a name for the trained model.
    --debug                         Enable debug routines. [default: False]
"""
import json
import os
import random
import pickle
from datetime import datetime

import git
import time
import logging
from typing import Dict, Any, Tuple
import tensorflow.compat.v2 as tf

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import build_vocab_from_data_dir, get_minibatch_iterator, load_data_from_dir
from model import SyntacticModelv2, BaseModel, SyntacticModelv1, SyntacticModelv3
from compute_training_data import training_dirs


def train(
        model: BaseModel,
        train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        valid_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int,
        max_epochs: int,
        patience: int,
        save_file: str,
):
    """
    :param train_data, valid_data: are tuples with (Nodes_tensorised, Actions_tensorised)
    """
    best_valid_loss, _ = model.run_one_epoch(
        get_minibatch_iterator(valid_data, batch_size, is_training=False),
        training=False,
    )
    logging.info(f"Initial valid loss: {best_valid_loss:.3f}.")
    model.save(save_file)
    best_valid_epoch = 0
    train_time_start = time.time()

    # TensorBoard
    log_dir = args['--tensorboard-logs-path'] + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer_train = tf.summary.create_file_writer(log_dir + "/train")
    writer_valid = tf.summary.create_file_writer(log_dir + "/valid")

    for epoch in range(1, max_epochs + 1):
        logging.info(f"== Epoch {epoch}")
        train_loss, train_acc = model.run_one_epoch(
            get_minibatch_iterator(train_data, batch_size, is_training=True),
            training=True,
        )
        logging.info(f" Train:  Loss {train_loss:.4f}, Acc {train_acc:.3f}")
        valid_loss, valid_acc = model.run_one_epoch(
            get_minibatch_iterator(valid_data, batch_size, is_training=False),
            training=False,
        )
        logging.info(f" Valid:  Loss {valid_loss:.4f}, Acc {valid_acc:.3f}")

        # Write to TensorBoard
        with writer_train.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('accuracy', train_acc, step=epoch)

        with writer_valid.as_default():
            tf.summary.scalar('loss', valid_loss, step=epoch)
            tf.summary.scalar('accuracy', valid_acc, step=epoch)

        # Save if good enough.
        if valid_loss < best_valid_loss:
            logging.info(
                f"  (Best epoch so far, loss decreased {valid_loss:.4f} from {best_valid_loss:.4f})",
            )
            model.save(save_file)
            logging.info(f"  (Saved model to {save_file})")
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            logging.info(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation loss.",
            )
            logging.info(
                f"Training took {total_time:.0f}s. Best validation loss: {best_valid_loss:.4f}",
            )
            break


def run(arguments) -> None:
    hyperparameters = BaseModel.get_default_hyperparameters()
    if args['--model'] == 'v1':
        hyperparameters = SyntacticModelv1.get_default_hyperparameters()
    elif args['--model'] == 'v2':
        hyperparameters = SyntacticModelv2.get_default_hyperparameters()
    elif args['--model'] == 'v3':
        hyperparameters = SyntacticModelv3.get_default_hyperparameters()

    hyperparameters["run_id"] = make_run_id(arguments)
    max_epochs = int(arguments.get("--max-num-epochs"))
    patience = int(arguments.get("--patience"))
    max_num_files = arguments.get("--max-num-files")

    # override hyperparams if flag is passed
    hypers_override = arguments.get("--hypers-override")
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))

    if args['--compute-data']:
        """
        Create and save the training data
        """
        #### Make list of all training data directories
        data_dirs = [os.path.join(args["--train-data-dir"], data_dir) for data_dir in training_dirs]

        logging.info("Computing data ...")
        vocab_nodes, vocab_actions = build_vocab_from_data_dir(
            data_dirs=data_dirs,
            vocab_size=hyperparameters["max_vocab_size"],
            max_num_files=max_num_files,
        )
        logging.info(f"  Built vocabulary of {len(vocab_actions)} entries.")
        all_nodes, all_actions, all_fathers = load_data_from_dir(
            vocab_nodes,
            vocab_actions,
            length=hyperparameters["max_seq_length"],
            data_dirs=data_dirs,
            max_num_files=max_num_files,
        )
        logging.info(f"  Built dataset of {all_nodes.shape[0]} training examples.")

        # Save data
        with open('data/vocab_nodes', 'wb') as output:
            pickle.dump(vocab_nodes, output)
        with open('data/vocab_actions', 'wb') as output:
            pickle.dump(vocab_actions, output)
        with open('data/all_nodes', 'wb') as output:
            pickle.dump(all_nodes, output)
        with open('data/all_actions', 'wb') as output:
            pickle.dump(all_actions, output)
        with open('data/all_fathers', 'wb') as output:
            pickle.dump(all_fathers, output)


        logging.info("Finished computing data ...")
        logging.info("Now exiting program. Rerun with loading the data from memory...")
        exit(0)

    logging.info("Loading data into memory...")
    ### Load data
    with open(os.path.join(args['--saved-data-dir'], 'vocab_nodes'), 'rb') as input:
        vocab_nodes = pickle.load(input)
    with open(os.path.join(args['--saved-data-dir'], 'vocab_actions'), 'rb') as input:
        vocab_actions = pickle.load(input)
    with open(os.path.join(args['--saved-data-dir'], 'all_nodes'), 'rb') as input:
        all_nodes = pickle.load(input)
    with open(os.path.join(args['--saved-data-dir'], 'all_actions'), 'rb') as input:
        all_actions = pickle.load(input)
    with open(os.path.join(args['--saved-data-dir'], 'all_fathers'), 'rb') as input:
        all_fathers = pickle.load(input)


    # Shuffle and split data into train/valid/test
    indices = np.arange(all_nodes.shape[0])
    np.random.shuffle(indices)
    ind_len = indices.shape[0]

    train_indices = indices[:(int)(0.8 * ind_len)]
    valid_indices = indices[(int)(0.8 * ind_len):(int)(0.9 * ind_len)]
    test_indices = indices[(int)(0.9 * ind_len):]

    # Make the datasets
    train_data = (all_nodes[train_indices],
                  all_actions[train_indices],
                  all_fathers[train_indices])
    valid_data = (all_nodes[valid_indices],
                  all_actions[valid_indices],
                  all_fathers[valid_indices])
    test_data = (all_nodes[test_indices],
                 all_actions[test_indices],
                 all_fathers[test_indices])

    logging.info(f"  Loaded {train_data[0].shape[0]} training samples.")
    logging.info(f"  Loaded {valid_data[0].shape[0]} validation samples.")

    # Construct model
    if args['--model'] == 'v1':
        model = SyntacticModelv1(hyperparameters, vocab_nodes, vocab_actions)
    elif args['--model'] == 'v2':
        model = SyntacticModelv2(hyperparameters, vocab_nodes, vocab_actions)
    elif args['--model'] == 'v3':
        model = SyntacticModelv3(hyperparameters, vocab_nodes, vocab_actions)
    model.build(([None, hyperparameters["max_seq_length"], 3]))
    logging.info("Constructed model, using the following hyperparameters:")
    logging.info(json.dumps(hyperparameters))

    # Peth for saving the model
    save_model_dir = args["--save-dir"]
    os.makedirs(save_model_dir, exist_ok=True)
    save_file = os.path.join(
        save_model_dir, f"{hyperparameters['run_id']}_best_model.bin"
    )

    train(
        model,
        train_data,
        valid_data,
        batch_size=hyperparameters["batch_size"],
        max_epochs=max_epochs,
        patience=patience,
        save_file=save_file,
    )


def make_run_id(arguments: Dict[str, Any]) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    user_save_name = arguments.get("--run-name")
    if user_save_name is not None:
        user_save_name = (
            user_save_name[: -len(".pkl")]
            if user_save_name.endswith(".pkl")
            else user_save_name
        )
        return "%s" % (user_save_name)
    else:
        return "RNNModel-%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))


if __name__ == "__main__":
    args = docopt(__doc__)

    # Set random seeds for reproducibility
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Logging configuration
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=args['--log-file'],
                        filemode='a')
    # define a handler which writes INFO messages or higher to the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))

    # add handler to the root logger
    logging.getLogger('').addHandler(console)

    # get current commit
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    logging.info(f"\n---Started Training from commit {sha}---\n")
    print(args)

    run_and_debug(lambda: run(args), args["--debug"])
