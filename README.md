Adaptation of the paper [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696) to code completion.

To visualise the AST of a .java file:
```
./scripts/sh_compile_and_visualise.sh -v -- ./test/Example.java
```

Tests/Debug:
```
python ./test/test_compute_action_sequence.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_tensorise_sequence.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_compute_grammar.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_compute_vocabulary.py --max-num-file=10 ./test
```

Compute data:
```
python train.py --compute-data\
                --saved-data-dir="./data"\
                --train-data-dir="../corpus-features"\
                --log-file="./logs/training.log"\
                --tensorboard-logs-path="./logs_tensorboard"\
                --max-num-files 50
```

Train:
```
python train.py --model='v2'\
                --save-dir="./trained_models"\
                --saved-data-dir="./data/250"\
                --log-file='./logs/training.log'\
                --tensorboard-logs-path="./logs_tensorboard"\
                --max-num-epochs 100\
                --patience 5
```

Hyper-parameter search:
```
python hyper_parameter_search.py --model='v2'\
                                  --save-dir="./trained_models"\
                                  --saved-data-dir="./data/250"\
                                  --log-file='./logs/training.log'\
                                  --log-file-hyperparams='./logs'\
                                  --tensorboard-logs-path="./logs_tensorboard"\
                                  --max-num-epochs 5\
                                  --patience 10
```

Evaluate:
```
python evaluate.py --trained-model="trained_models/RNNModel-2020-03-05-15-24-55_best_model.bin"\
                   --saved-data-dir="./data/250"\
                   --model="v1"\
                   --qualitative
```

Compute training data statistics:
```
python read_training_data.py --train-data-dir="../corpus-features"
```

Best models trained:
```
v1 
trained on 5000 trained files - RNNModel-2020-03-05-15-24-55_best_model.bin
train_data:  Loss 0.0043, Acc 0.827
valid_data:  Loss 0.0057, Acc 0.806
seen_test_data:  Loss 0.0056, Acc 0.808
unseen_test_data:  Loss 0.0066, Acc 0.778

v2
trained on 5000 trained files
```



# Wiki of internals
- A Node has two important fields: .type(Type of node in our own AST) and .contents(Java Symbol Type)  

# Special things in the AST
- Each variable in `int a, b, c` has a node `VARIABLE` that has an individual child `TYPE`, and all three type nodes are connected to the same `PRIMITIVE_TYPE` 