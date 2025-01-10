import numpy as np
import time 
import faiss 
import custom_invlists 

from faiss.contrib.datasets import SyntheticDataset 
from faiss.contrib.evaluation import RepeatTimer



def run(): 
    multi_thread = True

    if multi_thread: 
        ds = SyntheticDataset(32, 10_000, 1000_000, 10_000) 
    else: 
        ds = SyntheticDataset(32, 10_000, 1000_000, 1_000) 
    print(ds)
    
    index = faiss.index_factory(ds.d, "IVF1024,PQ4np")

    index.train(ds.get_train())
    index.add(ds.get_database())

    invlists_comp = {
        "ref": index.invlists,
        "packed-bits": custom_invlists.CompressedIDInvertedListsPackedBits(index.invlists),
        "elias-fano": custom_invlists.CompressedIDInvertedListsEliasFano(index.invlists),
        "roc": custom_invlists.CompressedIDInvertedListsFenwickTree(index.invlists), 
        "wavelet-tree": custom_invlists.CompressedIDInvertedListsWaveletTree(index.invlists),
        "wavelet-tree-1": custom_invlists.CompressedIDInvertedListsWaveletTree(index.invlists, 1)
    }

    invlists_comp["ref"].compressed_ids_size_in_bytes = 8 * index.ntotal
    index.own_invlists 
    index.own_invlists = False 
    k = 20

    if not multi_thread: 
        faiss.omp_set_num_threads(1)
        
    stats = faiss.cvar.indexIVF_stats
    for name, invlists in invlists_comp.items(): 
        # for implementations that offer random access
        decode_1by1 = name != "roc"
        print(f"invlist type {name} size {invlists.compressed_ids_size_in_bytes} {decode_1by1=}")
        index.replace_invlists(invlists, False)
        for nprobe in 1, 4, 16: 
            index.nprobe = nprobe 
            timer = RepeatTimer(warmup=1, runs=10, max_secs=10)
            index.parallel_mode = 0

            for _ in timer: 
                stats.reset()
                index.search(ds.get_queries(), k=k)

            print(f"immediate -- invlist {name} {nprobe=} time={timer.ms():.1f} ± {timer.ms_std():.1f} ms ({timer.nruns()} runs) ndis/q={stats.ndis/ds.nq:.3f} ")
            
            index.parallel_mode = 3

            timer = RepeatTimer(warmup=1, runs=10, max_secs=10)
            for _ in timer: 
                index.search_defer_id_decoding(ds.get_queries(), k=k, decode_1by1=decode_1by1)

            print(f"deferred  -- invlist {name} {nprobe=} time={timer.ms():.1f} ± {timer.ms_std():.1f} ms ({timer.nruns()} runs)")


run()

    
