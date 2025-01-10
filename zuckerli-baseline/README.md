To generate results for the Zuckerli baseline, first run the following the generate edgelists for all datasets (`.el` file extension).

```bash
    ./generate_graph_edgelists_launch.sh
```

Results are saved in `graphs/`.

Then, run


```bash
    ./compress_graphs_with_zuckerli.sh
```