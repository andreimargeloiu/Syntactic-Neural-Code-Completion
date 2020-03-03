"""
Usage:
    read_training_data.py [options]

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help                       Show this screen.
    --train-data-dir=NAME           Training directory path
"""
import os
from datetime import datetime

import numpy as np
from docopt import docopt

from dataset import get_data_files_from_directory, build_vocab_from_data_dir, load_data_from_dir

training_dirs = [
    'Bukkit',
    # 'cassandra',
    'clojure',
    # 'dubbo',
    # 'errorprone',
    'grails-core',
    'guice',
    # 'hibernate-orm',
    'jsoup',
    'junit4',
    # 'kafka',
    # 'libgdx',
    'metrics',
    'okhttp',
    # 'spring-framework',
    # 'tomcat'
]

# training_dirs = [
#     'jsoup'
# ]

if __name__ == "__main__":
    args = docopt(__doc__)

    train_data_dir = args['--train-data-dir']

    with open('./test_outputs/training_data_info.txt', 'w') as output:
        print("Loading data ...")
        print("%18s %11s %15s %15s %11s %15s" % ("Folder name", "File count", "Vocab actions", "Vocab nodes", "Methods", "Time_process"))
        start_time = datetime.now()
        for corpus_dir in training_dirs:
            corpus_path = os.path.join(train_data_dir, corpus_dir)

            file_count = len(get_data_files_from_directory(corpus_path))
            vocab_nodes, vocab_actions = build_vocab_from_data_dir(
                data_dir=corpus_path,
                vocab_size=1000,
                max_num_files=100,
            )

            train_data = load_data_from_dir(
                vocab_nodes,
                vocab_actions,
                length=50,
                data_dir=args["--train-data-dir"],
                max_num_files=1000,
            )
            # train_data = [np.zeros(2)]
            print("%15s %11s %15s %15s %11s %15s" % (corpus_dir, file_count,\
                                                     len(vocab_actions), len(vocab_nodes),\
                                                     train_data[0].shape[0], datetime.now() - start_time))




            # print(f"  Loaded {train_data[0].shape[0]} training samples from {args['--train-data-dir']}.")



