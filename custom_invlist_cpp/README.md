# What is this? 

This directory contains module `custom_invlists` that integrates with Faiss to compress the ids in IVF indexes. 

`custom_invlists` contains: 

- `search_IVF_defer_id_decoding`: an IVF search function that does not collect the result IDs immediately, but instead collects (invlist, offset) pairs and does the conversion to IDs when the search finished -- this avoids decompressing invlists without necessity

- four alternative `InvertedLists` objects that compress the IDs of the lists. The index itself does the vector compression. 

## How to compile 

The module depends on Faiss, SDSL and succint. 

See [here](../alt-graph-index) on how to install Faiss and SWIG. 

The [SDSL library](https://github.com/simongog/sdsl-lite), needs to be compiled and installed. 
To install, go to some directory and run
```
git clone https://github.com/simongog/sdsl-lite.git
cd sdsl-lite/
bash install.sh  $PWD/installed
```
then adjust the path `SDSL_PATH` in the Makefile 

The succint library, is a header-only library provided as a git submodule (make sure to checkout submodules).
succint does not need to be configured or compiled, so it's fine to skip the configuration with
```
cat > ../succinct/succinct_config.hpp << EOF
#pragma once
#define SUCCINCT_USE_LIBCXX 1
#define SUCCINCT_USE_INTRINSICS 1
#define SUCCINCT_USE_POPCNT 1
EOF 
```
With these three in place, just compile `custom_invlists` with make. 
This will also run a few sanity checks. 

