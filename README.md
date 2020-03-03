Adaptation of the paper [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696) to code completion.

To visualise the AST of a .java file:
```
.scripts/sh_compile_ans_visualise.sh -v -- ../test/Example.java
```

Tests/Debug:
```
python ./test/test_compute_action_sequence.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_tensorise_sequence.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_compute_grammar.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_compute_vocabulary.py --max-num-file=10 ./test
```

Train:
```
python train.py --save-dir="./trained_models"\
                --saved-data-dir="./data"\
                --train-data-dir="../corpus-features"\
                --log-file='./logs/training.log'\
                --tensorboard-logs-path="./logs_tensorboard"\
                --max-num-epochs 100\
                --patience 5\
                --max-num-files 250\
                --compute-data
```

Evaluate:
```
python evaluate.py --trained-model="trained_models/RNNModel-2020-02-27-07-48-44_best_model.bin"\
                   --test-dir="../corpus-features/jsoup/"
```

Compute training data statistics:
```
python read_training_data.py --train-data-dir="../corpus-features"
```



# Wiki of internals
- A Node has two important fields: .type(Type of node in our own AST) and .contents(Java Symbol Type)  

# Special things in the AST
- Each variable in `int a, b, c` has a node `VARIABLE` that has an individual child `TYPE`, and all three type nodes are connected to the same `PRIMITIVE_TYPE` 