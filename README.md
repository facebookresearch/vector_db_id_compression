
# ID compression for Vector databases 

This is the implementation of the paper [Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search](http://arxiv.org/pdf/fill_link_when_ready) by Daniel Severo, Matthew Muckley, Karen Ullrich, Giuseppe Ottaviano and Matthijs Douze. 

The package is implemented in Python and partly in C++.
The main package depends on the Elias-Fano implementation from the [Succint library](https://github.com/ot/succinct/blob/master/elias_fano.hpp) and the wavelet tree from [SDSL](https://github.com/simongog/sdsl-lite). 
The code for these libraries is included in the main tree XXXX TODO check if we use git subpackages 

## Compiling 

Most of the code is written as a plugin to the [Faiss](https://github.com/facebookresearch/faiss) vector search library. 
Therefore Faiss should be installed and available from Python.
We assume that Faiss is installed via Conda and that the Faiss headers are available in the `$CONDA_PREFIX/include` directory. 
We also assume that [swig](https://swig.org/) is installed (it is available in conda). 

The compilation is piloted via makefiles. 
Make should be run in 3 subdirectories: 
```


```

## Reproducing the results 

The results from table 1 are obtained with the command 

```

```

That should output [this log](link_to_a_gist).


## Baseline methods 





### Zuckerli 




## Citation 

If you use this package in a research work, please cite: 

```
@misc {


}
```

## License 

This package is provided under a [CC-by-NC license](https://creativecommons.org/licenses/by-nc/4.0/deed.en).
