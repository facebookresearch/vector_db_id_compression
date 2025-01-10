#!/bin/bash

# Convert graphs to zuckerli binary, skip if exists
for x in graphs/*.el; do
    if [[ ! -f "$x.bin" ]]
    then
        echo "converting $x"
        ./graph_utils/bin/gutil convert -i "$x" -F bin -o "$x.bin"
    fi
done

# Convert graphs to zuckerli binary, skip if exists
# PATH=$PATH:/private/home/mmuckley/projects/neuralcompression/code/llvm-project/build/bin
# conda activate zuckerli_build
for x in graphs/*.el.bin; do
    echo
    echo "Compressing $x"
    /private/home/dsevero/repos/zuckerli/build/encoder --input_path "$x" --output_path "$x.comp"
done