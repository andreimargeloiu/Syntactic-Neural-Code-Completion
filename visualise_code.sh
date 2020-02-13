#!/bin/bash

# Usage: gen_ast.sh Test.java

verbose=""
while [ -n "$1" ]; do # while loop starts
    case "$1" in

        -v) verbose="--verbose-dot" ;; # Message for -a option

        --)
            shift # The double dash which separates options from parameters

            break
            ;; # Exit the loop using break command

        *) echo "Option $1 not recognized" ;;

    esac

    shift

done
total=1
echo "$1"


# Run the feature extractor to produce the .proto file containing the features extracted from the program
javac -cp "$PWD"/extractor/target/features-javac-extractor-1.0.0-SNAPSHOT-jar-with-dependencies.jar -Xplugin:FeaturePlugin "$1"

# Convert the .proto file into a Graphviz file
java -jar "$PWD"/dot/target/features-javac-dot-1.0.0-SNAPSHOT-jar-with-dependencies.jar -i "$1".proto -o "$1".dot "$verbose"

dot -Tpng "$1".dot > "$1".png

rm "$1".dot

open "$1".png