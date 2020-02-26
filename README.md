Adaptation of the paper [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696) to code completion.

To visualise the AST of a .java file:
```
.scripts/sh_compile_ans_visualise.sh -v -- ../test/Example.java
```

Tests/Debug:
```
python ./test/test_compute_action_sequence.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_tensorise_sequence.py --max-num-file=10 ../../corpus-features/jsoup/
python ./test/test_compute_grammar --max-num-file=10 ../../corpus-features/jsoup/
```

To train the model:
```
python train.py --log-path="./logs/training.log" --save-dir="./trained_models" --train-data-dir="../corpus-features/jsoup" --valid-data-dir="../corpus-features/jsoup/"
```

# Wiki of internals
- A Node has two important fields: .type(Type of node in our own AST) and .contents(Java Symbol Type)  

# Special things in the AST
- Each variable in `int a, b, c` has a node `VARIABLE` that has an individual child `TYPE`, and all three type nodes are connected to the same `PRIMITIVE_TYPE` 