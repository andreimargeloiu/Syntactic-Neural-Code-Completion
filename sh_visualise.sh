# Convert the .proto file into a Graphviz file
echo "$1"

java -jar "$PWD"/dot/target/features-javac-dot-1.0.0-SNAPSHOT-jar-with-dependencies.jar -i "$1" -o "$1".dot "--verbose-dot"

dot -Tpng "$1".dot > "$1".png

rm "$1".dot

open "$1".png