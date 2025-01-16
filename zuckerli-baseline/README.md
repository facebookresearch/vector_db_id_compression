# Zuckerli baseline

## 1) Generate `.el` file for each dataset
Run `generate_graph_edelists.py` to generate an `.el` file for some dataset and index string.
This script takes 3 arguments:
- Dataset index, specifying which dataset to generate the `.el` for. See the `AVAILABLE_DATASETS` variable.
- Max degree parameter of NSG and HNSW.
- Path to the `fb_ssnpp/` directory (only used if dataset index is set to `3`).

To generate `.el` files for all datasets in the paper, run the following

```bash
fb_ssnpp_dir=...
for dataset_idx in 0 1 2 3; do
    for max_degree in 16 32 64 128 256; do
        ./generate_graph_edgelists_sbatch.sh $dataset_idx $max_degree $fb_ssnpp_dir
    done
done
```

`.el` files will be saved in `graphs/`.

## 2) Install veluca93/graph_utils and generate Zuckerli graph binaries

Compile the `gutil` following instructions in https://github.com/veluca93/graph_utils/tree/a9521a943f67466e0e1badaf10876e82c2fbef2a

Create a `.bin` file for each graph

```sh
# Convert graphs to zuckerli binary, skip if exists
for x in graphs/*.el; do
    if [[ ! -f "$x.bin" ]]
    then
        echo "converting $x"
        ./graph_utils/bin/gutil convert -i "$x" -F bin -o "$x.bin"
    fi
done
```

## 3) Compile Zuckerli and compress

Compile Zuckerli from https://github.com/google/zuckerli/tree/874ac40705d1e67d2ee177865af4a41b5bc2b250

Assuming you cloned `zuckerli` in the local directory, run the following to compress each graph.
```sh
for x in graphs/*.el.bin; do
    zuckerli/build/encoder --input_path "$x" --output_path "$x.comp"
done
```

This will generate logs similar to https://gist.github.com/dsevero/ac356c1c1cdf4aac17eee34387a5a4b2