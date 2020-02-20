Adaptation of the paper [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696) to code completion.

To visualise the AST of a .java file:
```
./visualise_code.sh -v -- ./test_java_code/Example.java
```

To compute the action sequence:
```
python test_compute_action_sequence.py --max-num-file=10 ../corpus-features/jsoup/
```

To test the tensorised token sequence:
```
python test_tensorise_sequence.py --max-num-file=10 ../corpus-features/jsoup/
```


# Wiki of internals
- A Node has two important fields: .type(Type of node in our own AST) and .contents(Java Symbol Type)  

# Special things in the AST
- Each variable in `int a, b, c` has a node `VARIABLE` that has an individual child `TYPE`, and all three type nodes are connected to the same `PRIMITIVE_TYPE` 