for index in PQ8x10np; do
    for num_clusters in 1024; do
        for dataset in 3; do
            sbatch benchmark-slurm.sh $index $dataset $num_clusters
        done
    done
done

for index in Flat; do
    for num_clusters in 256; do
        for dataset in 3; do
            sbatch benchmark-slurm.sh $index $dataset $num_clusters
        done
    done
done