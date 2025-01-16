# What is this? 

This directory contains additional module `altid` to the Faiss indexes useful for ID compression of NSG indexes. 

`altid` contains 3 functions for alternative id storage: 

- a function that traces the nodes visited during NSG search
- three classes that contain the graph structure (alternatives of the regular uncompressed one)

## How to compile 

Note that this module needs a Faiss compiled after 2024-10-22 to compile (because replacing the NSG graph was not supported before).
To install this Faiss version on conda, do: 

```
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.9.0 pytorch=*=*cuda*
```

This also installs pytorch in case you'd need it... 
To compile, just run `make`.
This automatically runs the test below. 

## How to test 

Run 
```
python -m unittest test_altid.py
```
See also that file on how to use the module. 

## How to modify

The C++ implementation is in `altid_impl.{h,cpp}`. 
