# ID compression for Vector databases 

This is the implementation of the paper [Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search](http://arxiv.org/abs/2501.10479) by Daniel Severo, Giuseppe Ottaviano, Matthew Muckley, Karen Ullrich, and Matthijs Douze. 

The package is implemented in Python and partly in C++.
The main package depends on the Elias-Fano implementation from the [Succint library](https://github.com/ot/succinct/blob/master/elias_fano.hpp) and the wavelet tree from [SDSL](https://github.com/simongog/sdsl-lite). 
The code for these libraries is included as git submodules. 

## TABLE OF CONTENTS

- [1) Compiling and Installing Dependencies](#1-compiling-and-installing-dependencies)
- [2) Reproducing results in the paper](#2-reproducing-results-in-the-paper)
  - [2.1) Tables 1 and 2 (online setting)](#21-tables-1-and-2-online-setting)
    - [Graph indices](#graph-indices)
    - [IVF Indices](#ivf-indices)
  - [2.2) Table 3 (offline setting)](#22-table-3-offline-setting)
    - [Random Edge Coding (REC)](#random-edge-coding-rec)
    - [Zuckerli Baseline](#zuckerli-baseline)
  - [2.3) Table 4 (Large-scale experiment with QINCo)](#23-table-4-large-scale-experiment-with-qinco)
- [3) Citation](#3-citation)
- [4) License](#4-license)

## 1) Compiling and Installing Dependencies

Most of the code is written as a plugin to the [Faiss](https://github.com/facebookresearch/faiss) vector search library. 
Therefore Faiss should be installed and available from Python.
We assume that Faiss is installed via Conda and that the Faiss headers are available in the `$CONDA_PREFIX/include` directory. 
We also assume that [swig](https://swig.org/) is installed (it is available in conda). 

The compilation is piloted via makefiles (that are written for Linux). 
Make should be run in the two subdirectories [alt-graph-index](./alt-graph-index) and [custom_invlist_cpp](./custom_invlist_cpp), see compilation instructions there.


To complete the setup, do the following.
- Create a conda environment with python 3.10. We suggest using 
```sh
conda create -n vector_db_id_compression python=3.10
```
- Activate your environment (e.g., `conda activate vector_db_id_compression`)
- Install [Faiss](https://github.com/facebookresearch/faiss).
- Install external dependencies (including Python dependencies) by running `/install-dependencies.sh`

## 2) Reproducing results in the paper

### 2.1) Tables 1 and 2 (online setting)

#### Graph indices
To reproduce graph-index results of Table 1, first make sure you have installed `succinct` as specified in [succinct/README.md](https://github.com/ot/succinct/blob/669eebbdcaa0562028a22cb7c877e512e4f1210b/README.md)

Then, install the compressed graph indices by running `make` in the `alt-graph-index` directory.

From `alt-graph-index/`, run `graph_dynamic_bench_invlists.py`.
This script takes 3 arguments:

- Dataset index, specifying which dataset to generate the .el for. See the AVAILABLE_DATASETS variable.
- Max degree parameter of NSG.
- Path to the `fb_ssnpp/` directory (only used if dataset index is set to 3).

To reproduce all graph results in Table 1, run the following code.

```sh
cd alt-graph-index
fb_ssnpp_dir=...
for dataset_idx in 0 1 2 3; do
    for max_degree in 16 32 64 128 256; do
        python graph_dynamic_bench_invlists.py $dataset_idx $max_degree $fb_ssnpp_dir
    done
done
```

A CSV file with results will be saved to `alt-graph-index/results-online-graphs`, for each run.

---

#### IVF Indices 

To reproduce IVF index results of Table 1, first install the compressed IVF indices by running `make` in the `custom_invlist_cpp` directory.

From the `custom_invlist_cpp` directory run `bench_invlists.py`.
This script takes 3 arguments:

- Dataset index. See the AVAILABLE_DATASETS variable.
- Index string in the style of the [FAISS index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory).
- Path to the `fb_ssnpp/` directory (only used if dataset index is set to 3).

To reproduce all IVF results in Table 1, run the following code.
```sh
cd custom_invlist_cpp
fb_ssnpp_dir=...
for dataset_idx in 0 1 2 3; do
    for code in Flat PQ4 PQ16 PQ32 PQ8x10; do
        for num_clusters in 256 512 1024 2048; do
            index_str=IVF$num_clusters,$code
            python bench_invlists.py $dataset_idx $index_str $fb_ssnpp_dir
        done
    done
done
```

A CSV file with results will be saved to `custom_invlist_cpp/results-online-ivf`, for each run.

### 2.2) Table 3 (offline setting)

#### Random Edge Coding (REC)
For these experiments you will need to clone [dsevero/Random-Edge-Coding](https://github.com/dsevero/Random-Edge-Coding) into the root directory. Follow the instructions in [Random-Edge-Coding/README.md](https://github.com/dsevero/Random-Edge-Coding?tab=readme-ov-file#how-to-use-random-edge-coding) to install REC.

Note: assuming you cloned `facebookresearch/vector_db_id_compression` (i.e., this repo) with `git clone --recursive`, then REC will automatically be cloned as well.

#### Zuckerli Baseline
See [zuckerli-baseline/README.md](zuckerli-baseline/README.md) for the baseline.

### 2.3) Table 4: Large-scale experiment with QINCo

The [QINCo](https://github.com/facebookresearch/Qinco/tree/main/qinco_v1) package should be available as a subdirectory of `custom_invlists_cpp`: 
```sh
cd custom_invlists_cpp
git clone https://github.com/facebookresearch/Qinco.git
mv Qinco/qinco_v1 .    # the code of the original qinco 
rm -rf Qinco
mv qinco_v1 Qinco
```

A small scale validation experiment is with 10M vectors, using a pre-built QINCo index

<details><summary>commands</summary>
  
```sh
tmpdir=/scratch/matthijs/  # some temporary directory

# data from https://github.com/facebookresearch/Qinco/blob/main/docs/IVF_search.md
(cd $tmpdir; wget https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF65k_16x8_L2.pt )
(cd $tmpdir ; wget https://dl.fbaipublicfiles.com/QINCo/ivf/bigann10M_IVF65k_16x8_L2.faissindex )

# run baseline without id compression
# parameters are one of the optimal op points from 
# https://gist.github.com/mdouze/e4b7c9dbf6a52e0f7cf100ce0096aaa8
# cno=21

#  baseline 

python search_ivf_qinco.py --db bigann10M \
 --model $tmpdir/bigann_IVF65k_16x8_L2.pt \
 --index $tmpdir/bigann10M_IVF65k_16x8_L2.faissindex \
 --todo search  --nthread 32 --nprobe 64 --nshort 100 

# with ROC 

python search_ivf_qinco.py --db bigann10M \
 --model $tmpdir/bigann_IVF65k_16x8_L2.pt \
 --index $tmpdir/bigann10M_IVF65k_16x8_L2.faissindex \
 --todo search  --nthread 32 --nprobe 64 --nshort 100  \
 --id_compression roc --defer_id_decoding

```

</details>

This should produce an output similar to [this log](https://gist.github.com/mdouze/b28e1172f612764dc2cf5133b5614f7d)

Note that `search_ivf_qinco.py` is a slightly adapted version of QINCo's [`search_ivf.py`](https://github.com/facebookresearch/Qinco/blob/main/qinco_v1/search_ivf.py) to integrate id compression. 

#### Full scale experiment 

Similar to above (but slower!): 

<details><summary>commands</summary>

```sh

datadir=/checkpoint/matthijs/id_compression/qinco/

(cd $datadir && wget https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_8x8_L2.faissindex)
(cd $datadir && wget https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF1M_8x8_L2.pt)

# cno=533 from https://gist.github.com/mdouze/0187c2ca3f96f806e41567af13f80442
# fastest run with R@1 > 0.3
params="--nprobe 128 --quantizer_efSearch 64 --nshort 200"

for comp in none packed-bits elias-fano roc wavelet-tree wavelet-tree-1; do 

    python -u search_ivf_qinco.py \
        --todo search \
        --db bigann1B \
        --model $datadir/bigann_IVF1M_8x8_L2.pt \
        --index $datadir/bigann1B_IVF1M_8x8_L2.faissindex \
        --nthread 32 \
        --id_compression $comp --defer_id_decoding --redo_search 10 \
        $params

done 

```

</details>

This outputs [these logs](https://gist.github.com/mdouze/93491e398da661843f215b17525eda59).
The 10 runs are averaged to produce table 4 using 
[parse_qinco_res.py](https://gist.github.com/mdouze/8fe85335197049db4d728ae0b427036f).

## 3) Citation 

If you use this package in a research work, please cite: 

```
@misc{severo2025idcompression,
   title="Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search",
   author={Daniel Severo and Giuseppe Ottaviano and  Matthew Muckley and Karen Ullrich and Matthijs Douze},
   year={2025},
   eprint={2501.10479},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```

## 4) License 

This package is provided under a [CC-by-NC license](https://creativecommons.org/licenses/by-nc/4.0/deed.en).
