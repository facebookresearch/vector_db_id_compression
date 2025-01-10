#pragma once 

#include <faiss/IndexIVF.h>
#include <faiss/IndexNSG.h>
#include "../succinct/elias_fano.hpp"
#include "../fenwick_tree_cpp/src/fenwick_tree.h"
#include "../custom_invlist_cpp/codec.h"



/* perform a search in the NSG graph and collect the ids of the nodes that are compared */
void search_NSG_and_trace(
        const faiss::IndexNSG & index, 
        faiss::idx_t n, 
        const float *x, 
        int k, 
        faiss::idx_t *labels, 
        float *distances, 
        std::vector<faiss::idx_t> & visited_nodes);


/**  NSG graph where edges are encoded into the minimum number of bits */
struct CompactBitNSGGraph : faiss::nsg::Graph<int32_t> {
    int bits;        ///< number of bits per edge
    size_t stride;   ///< number of bytes per node 

    ///< array of size N * stride 
    std::vector<uint8_t> compressed_data;

    CompactBitNSGGraph(const faiss::nsg::Graph<int32_t>& graph);

    size_t get_neighbors(int i, int32_t* neighbors) const override;
};

/**  NSG graph where edges are encoded using Elias-Fano */
struct EliasFanoNSGGraph : faiss::nsg::Graph<int32_t> {
    std::vector<succinct::elias_fano*> ef_bitstreams;

    size_t compressed_ids_size_in_bytes = 0;
    size_t overhead_in_bytes = 0;
    EliasFanoNSGGraph(const faiss::nsg::Graph<int32_t>& graph);

    size_t get_neighbors(int i, int32_t* neighbors) const override;
};


using ROCSymbolType = int32_t;

/**  NSG graph where edges are encoded using ROC */
struct ROCNSGGraph : faiss::nsg::Graph<int32_t> {
    std::vector<std::vector<uint8_t>> codes_all; 
    std::vector<ANSState> ans_states;    
    std::vector<uint64_t> id_symbol_precision;
    size_t compressed_ids_size_in_bytes = 0;
    std::vector<uint32_t> num_outgoing_edges;
    size_t overhead_in_bytes = 0;

    ROCNSGGraph(const faiss::nsg::Graph<int32_t>& graph);

    size_t get_neighbors(int i, int32_t* neighbors) const override;
};