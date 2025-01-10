for m in 16 32 64 128 256; do
    for d in 0 1 2 3; do
        sbatch generate_graph_edgelists_sbatch.sh $d $m
    done
done