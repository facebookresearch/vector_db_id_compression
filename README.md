
# IVF compression

## IVF compression in Faiss

Simplified version of a uniform codec without replacement: simplified_compression.py

Re-written in C++: codec.cpp / codec.h

Use this to implement a [custom InvertedLists](https://github.com/facebookresearch/faiss/wiki/Inverted-list-objects-and-scanners#the-invertedlists-object) object: custom_invlists.swig
See comments in that file on how to compile. 

Benchmark search with this: 

```
(faiss_1.8.0) matthijs@matthijs-mbp intern_faiss_ivf_compression % python bench_invlists.py
train
WARNING clustering 20000 points to 1024 centroids: please provide at least 39936 training points
add
ref search time: 0.103 s
convert invlists
new search time: 0.911 s
```

## Run IVF benchmark
```bash
cd custom_invlist_cpp
make clean && make
python bench_invlists.py
```