// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "altid_impl.h"

#include <faiss/utils/hamming.h>
#include <faiss/impl/FaissAssert.h>


/********************************************** NSG extensions */


using namespace faiss; 

using FinalNSGGraph = nsg::Graph<int32_t>;

CompactBitNSGGraph::CompactBitNSGGraph(const FinalNSGGraph& graph)
        : FinalNSGGraph(graph.data, graph.N, graph.K) {
    bits = 0; 
    while((1 << bits) < N + 1) bits++; 
    stride = (K * bits + 7) / 8;
    compressed_data.resize(N * stride);
    for (size_t i = 0; i < N; i++) {
        BitstringWriter writer(compressed_data.data() + i * stride, stride);
        for (size_t j = 0; j < K; j++) {
            int32_t v = graph.data[i * K + j];
            if (v == -1) {
                writer.write(N, bits);
                break;
            } else {
                writer.write(v, bits);
            }
        }
    }
    data = nullptr;
}

size_t CompactBitNSGGraph::get_neighbors(int i, int32_t* neighbors) const {
    BitstringReader reader(compressed_data.data() + i * stride, stride);
    for (int j = 0; j < K; j++) {
        int32_t v = reader.read(bits);
        if (v == N) {
            return j;
        }
        neighbors[j] = v;
    }
    return K;
}

EliasFanoNSGGraph::EliasFanoNSGGraph(const FinalNSGGraph& graph)
        : FinalNSGGraph(graph.data, graph.N, graph.K) {
    std::vector<uint32_t> num_outgoing_edges(N, 0);
    overhead_in_bytes += N*std::ceil(std::log2(N)) / 8.0; // size of each set (i.e., friendlist)
    overhead_in_bytes += N*std::ceil(std::log2(N)) / 8.0; // max_id value
    for (size_t i = 0; i < N; i++) {

        // Compute number of outgoing edges
        for (size_t j = 0; j < K; j++) {
            int32_t v = graph.data[i * K + j];
            if (v == -1) {
                break;
            } else {
                num_outgoing_edges[i]++;
            }
        }

        // Compress ids with Elias-Fano
        uint32_t ls = num_outgoing_edges[i];
        int32_t* start_friendlist = graph.data + i * K; 
        int32_t* end_friendlist = start_friendlist + ls; 

        int32_t max_id = *std::max_element(start_friendlist, end_friendlist);
        std::sort(start_friendlist, end_friendlist);
        succinct::elias_fano::elias_fano_builder ef_builder(max_id, ls);
        for (size_t j=0; j < ls; j++)
        {
            int32_t v = graph.data[i * K + j];
            if (v < 0) break;
            ef_builder.push_back((uint64_t)v);
        }
        succinct::elias_fano* ef_ptr = new succinct::elias_fano(&ef_builder);
        ef_bitstreams.push_back(ef_ptr);
        compressed_ids_size_in_bytes += ef_ptr->m_low_bits.size() + ef_ptr->m_high_bits.size();
    }
    compressed_ids_size_in_bytes /= 8;
    data = nullptr;
}

size_t EliasFanoNSGGraph::get_neighbors(int i, int32_t* neighbors) const {
    const auto& ef = ef_bitstreams[i];
    succinct::elias_fano::select_enumerator it(*ef, 0); 

    for (size_t i=0; i < ef->num_elements; i++) {
        neighbors[i] = it.next();
    }

    return ef->num_elements; 
}

ROCNSGGraph::ROCNSGGraph(const FinalNSGGraph& graph)
        : FinalNSGGraph(graph.data, graph.N, graph.K) {
    num_outgoing_edges.resize(N);
    overhead_in_bytes += N*std::ceil(std::log2(N)) / 8.0;
    ans_states.resize(graph.N);
    for (size_t list_no = 0; list_no < graph.N; list_no++) {
        // Compute number of outgoing edges
        for (size_t j = 0; j < K; j++) {
            int32_t v = graph.data[list_no * K + j];
            if (v == -1) {
                break;
            } else {
                num_outgoing_edges[list_no]++;
            }
        }

        // Compress with ROC
        uint32_t ls = num_outgoing_edges[list_no];
        int32_t* start_friendlist = graph.data + list_no * K; 
        int32_t* end_friendlist = start_friendlist + ls; 

        int32_t max_id = *std::max_element(start_friendlist, end_friendlist);
        id_symbol_precision.push_back((uint64_t)std::ceil(std::log2(max_id)));

        // Populate fenwick tree in random order to increase balancedness
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(start_friendlist, end_friendlist, g);

        FenwickTree<ROCSymbolType> ftree;
        for (int i=0; i < ls; i++) {
            ROCSymbolType symbol = start_friendlist[i];
            ftree.insert_then_forward_lookup(symbol);
        }

        for (size_t i = 0; i < ls; i++) {
            // Sample, without replacement, an element from the FenwickTree.
            uint32_t nmax = ls - i;
            size_t index = pop_with_finer_precision(ans_states[list_no], nmax);
            Range range = ftree.reverse_lookup_then_remove(index);

            // Encode id.
            ROCSymbolType id = range.ftree->symbol;
            codec_push(ans_states[list_no], id, id_symbol_precision[list_no]);
        }
        compressed_ids_size_in_bytes += ans_states[list_no].size();
    }
    data = nullptr;
}

size_t ROCNSGGraph::get_neighbors(int node, int32_t* neighbors) const {
    FenwickTree<ROCSymbolType> ftree;
    ANSState state(ans_states[node]);
    uint32_t n = num_outgoing_edges[node]; 
    for (size_t i = 0; i < n; i++) {
        int32_t symbol = (uint64_t)codec_pop(state, id_symbol_precision[node]);
        auto range = ftree.insert_then_forward_lookup(symbol);
        uint32_t nmax = i + 1;
        push_with_finer_precision(state, range.start, nmax);
        neighbors[n - i - 1] = range.ftree->symbol;
    }
    return K;
}


namespace {

struct TracingDistanceComputer: DistanceComputer {
    std::vector<idx_t> visited; 
    DistanceComputer *basedis; 

    TracingDistanceComputer(DistanceComputer *basedis): 
        basedis(basedis) {}
    
    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        visited.push_back(i); 
        return (*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        visited.push_back(i); 
        visited.push_back(j); 
        return basedis->symmetric_dis(i, j);
    }

    virtual ~TracingDistanceComputer() {
        delete basedis;
    }

};

} // anonymous namespace 


void search_NSG_and_trace(
        const IndexNSG & index, 
        idx_t n, 
        const float *x, 
        int k, 
        faiss::idx_t *labels, 
        float *distances, 
        std::vector<idx_t> & visited_nodes) {

    int L = std::max(index.nsg.search_L, (int)k); // in case of search L = -1

    VisitedTable vt(index.ntotal);

    std::unique_ptr<TracingDistanceComputer> dis(
        new TracingDistanceComputer(
            nsg::storage_distance_computer(index.storage)));

    for (idx_t i = 0; i < n; i++) {
        idx_t* idxi = labels + i * k;
        float* simi = distances + i * k;
        dis->set_query(x + i * index.d);

        index.nsg.search(*dis, k, idxi, simi, vt);

        vt.advance();
    }

    std::swap(visited_nodes, dis->visited); 
}
