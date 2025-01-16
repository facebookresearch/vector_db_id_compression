// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "custom_invlists_impl.h"

#include <unordered_map>
#include <faiss/impl/FaissAssert.h>

#include "../fenwick_tree_cpp/src/fenwick_tree.h"


/********************************************************************************* 
 *  InvertedListsArrayCodes
 *********************************************************************************/ 

InvertedListsArrayCodes::InvertedListsArrayCodes(const faiss::InvertedLists & il): 
    ReadOnlyInvertedLists(il.nlist, il.code_size) {}


size_t InvertedListsArrayCodes::list_size(size_t list_no) const  {
    return codes_all[list_no].size() / code_size; 
}

const uint8_t* InvertedListsArrayCodes::get_codes(size_t list_no) const  {
    return codes_all[list_no].data(); 
}

/********************************************************************************* 
 *  Packed bits 
 *********************************************************************************/ 

uint64_t BitstringReader_get_bits(const faiss::BitstringReader &bs, size_t i, int nbit) {
    assert(bs.code_size * 8 >= nbit + i);
    // nb of available bits in i / 8
    int na = 8 - (i & 7);
    // get available bits in current byte
    uint64_t res = bs.code[i >> 3] >> (i & 7);
    if (nbit <= na) {
        res &= (1 << nbit) - 1;
        return res;
    } else {
        int ofs = na;
        size_t j = (i >> 3) + 1;
        nbit -= na;
        while (nbit > 8) {
            res |= ((uint64_t)bs.code[j++]) << ofs;
            ofs += 8;
            nbit -= 8; // TODO remove nbit
        }
        uint64_t last_byte = bs.code[j];
        last_byte &= (1 << nbit) - 1;
        res |= last_byte << ofs;
        return res;
    }
}

using idx_t = faiss::idx_t; 



CompressedIDInvertedListsPackedBits::CompressedIDInvertedListsPackedBits(const faiss::InvertedLists & il): 
    InvertedListsArrayCodes(il)
{
    
    bits = 0; 
    size_t ntotal = il.compute_ntotal();
    while((1 << bits) < ntotal + 1) bits++; 
    codes_all.resize(nlist); 
    ids_all.resize(nlist);
    compressed_ids_size_in_bytes = 0;
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t ls = il.list_size(list_no); // number of elements in voronoi cell
        ScopedIds ids_in(&il, list_no);
        ScopedCodes codes(&il, list_no);

        const uint64_t* ids_data = (const uint64_t*)ids_in.get();
        const uint8_t* codes_data = (const uint8_t*)codes.get();

        ids_all[list_no].resize((ls * bits + 7) / 8); 
        faiss::BitstringWriter bs(ids_all[list_no].data(), ids_all[list_no].size());
        compressed_ids_size_in_bytes += ids_all[list_no].size();

        for(size_t i = 0; i < ls; i++) {
            FAISS_THROW_IF_NOT(ids_in[i] >= 0 && ids_in[i] < ntotal);
            bs.write(ids_in[i], bits);
        }

        codes_all[list_no].resize(code_size * ls);
        memcpy(codes_all[list_no].data(), codes.get(), codes_all[list_no].size()); 
    }
}

const faiss::idx_t* CompressedIDInvertedListsPackedBits::get_ids(size_t list_no) const  {
    size_t ls = list_size(list_no); 
    idx_t *the_ids = new idx_t[ls]; 
    faiss::BitstringReader bs(ids_all[list_no].data(), ids_all[list_no].size());
    
    for(size_t i = 0; i < ls; i++) {
        the_ids[i] = bs.read(bits);
    }
    return the_ids;        
}


faiss::idx_t CompressedIDInvertedListsPackedBits::get_single_id(size_t list_no, size_t offset) const {
    size_t ls = list_size(list_no); 
    faiss::BitstringReader bs(ids_all[list_no].data(), ids_all[list_no].size());
    return BitstringReader_get_bits(bs, offset * bits, bits);

}


void CompressedIDInvertedListsPackedBits::release_ids(size_t list_no, const idx_t* ids_in) const  {
    delete [] ids_in; 
}


namespace {
/*
void get_sorted_invlist(
    const faiss::InvertedLists & il, size_t list_no, 
)
*/
} // anonymous namespace 

/********************************************************************************* 
 *  Fenwick Tree
 *********************************************************************************/ 

CompressedIDInvertedListsFenwickTree::CompressedIDInvertedListsFenwickTree(const faiss::InvertedLists & il): 
    InvertedListsArrayCodes(il)
{

    // First element is the id value, second element is a pointer to a code.
    // Note, however, that only the ids are compressed into the ANS state.
    // The FenwickTree is over this tuple datatype only to facilitate reordering of the codes at compression time.
    using SymbolType = std::tuple<uint64_t,const uint8_t*>;

    ans_states.resize(nlist);  // nlist = number of voronoi cells
    codes_all.resize(nlist); 
    id_symbol_precision.resize(nlist);


#pragma omp parallel for
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t ls = il.list_size(list_no); // number of elements in voronoi cell
        std::vector<uint8_t> codes_reordered;

        if (ls == 0) {
            continue;
        }

        ScopedIds ids_in(&il, list_no);
        ScopedCodes codes(&il, list_no);

        const uint64_t* ids_data = (const uint64_t*)ids_in.get();
        const uint8_t* codes_data = (const uint8_t*)codes.get();
        FenwickTree<SymbolType> ftree;

        int max_id = *std::max_element(ids_data, ids_data + ls);
        id_symbol_precision[list_no] = (uint64_t)std::ceil(std::log2(max_id)); 

        // Populate fenwick tree in random order to increase balancedness
        std::vector<int> indices(ls);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int i : indices) {
            SymbolType symbol = std::make_tuple(ids_data[i], codes_data + i * code_size);
            ftree.insert_then_forward_lookup(symbol);
        }

        for (size_t i = 0; i < ls; i++) {
            // Sample, without replacement, an element from the FenwickTree.
            uint32_t nmax = ls - i;
            size_t index = pop_with_finer_precision(ans_states[list_no], nmax);
            Range range = ftree.reverse_lookup_then_remove(index);

            // Encode id.
            uint64_t id = std::get<0>(range.ftree->symbol);
            codec_push(ans_states[list_no], id, id_symbol_precision[list_no]);

            // Encode codes (no compression atm).
            const uint8_t* p = std::get<1>(range.ftree->symbol);
            codes_reordered.insert(codes_reordered.end(), p, p + code_size);

        }
        codes_all[list_no] = codes_reordered;
    }

    compressed_ids_size_in_bytes = 0;
    codes_size_in_bytes = 0;
    for (size_t list_no = 0; list_no < il.nlist; list_no++) {
        if (list_size(list_no) == 0) {
            continue; // let's prtend no memory is used here 
        }
        compressed_ids_size_in_bytes += ans_states[list_no].size();
        for (const auto& codes_this_voronoi_cell : codes_all) {
            codes_size_in_bytes += codes_this_voronoi_cell.size() * sizeof(uint8_t);
        }
    }

}

const faiss::idx_t* CompressedIDInvertedListsFenwickTree::get_ids(size_t list_no) const  {
    size_t ls = list_size(list_no); 
    if (ls == 0) {
        return nullptr;
    }
    idx_t *the_ids = new idx_t[ls]; 
    ANSState copy(ans_states[list_no]);
    decompress(copy, ls, (uint64_t*)the_ids, id_symbol_precision[list_no]); 
    return the_ids;        
}

void CompressedIDInvertedListsFenwickTree::release_ids(size_t list_no, const idx_t* ids_in) const  {
    delete [] ids_in; 
}

/********************************************************************************* 
 *  Elias-Fano
 *********************************************************************************/ 

CompressedIDInvertedListsEliasFano::CompressedIDInvertedListsEliasFano(const faiss::InvertedLists & il): 
    InvertedListsArrayCodes(il) 
{
    codes_all.resize(nlist);
    ef_bitstreams.resize(nlist); 
#pragma omp parallel for 
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t ls = il.list_size(list_no); // number of elements in voronoi cell

        if (ls == 0) {
            continue; 
        }

        std::vector<uint8_t> codes_reordered;
        std::vector<uint64_t> ids_reordered;

        ScopedIds ids_in(&il, list_no);
        ScopedCodes codes(&il, list_no);

        const uint64_t* ids_data = (uint64_t*)ids_in.get();
        const uint8_t* codes_data = (uint8_t*)codes.get();

        // Sort codes and ids, together
        auto pairs = canonicalize_order_inplace(ids_data, codes_data, ls);
        for (size_t i = 0; i < ls; ++i) {
            ids_reordered.push_back(pairs[i].first);
            const uint8_t* p = pairs[i].second;
            codes_reordered.insert(codes_reordered.end(), p, p + code_size);
        }

        codes_all[list_no] = codes_reordered;

        // Compress ids with Elias-Fano
        uint64_t max_id = *std::max_element(ids_reordered.begin(), ids_reordered.end());
        succinct::elias_fano::elias_fano_builder ef_builder(max_id, ls);
        for (size_t i=0; i < ls; i++)
        {
            ef_builder.push_back(ids_reordered[i]);
        }
        succinct::elias_fano* ef_ptr = new succinct::elias_fano(&ef_builder);
        ef_bitstreams[list_no] = ef_ptr;
    }

    compressed_ids_size_in_bytes = 0;
    codes_size_in_bytes = 0;
    for (size_t list_no = 0; list_no < il.nlist; list_no++) {
        succinct::elias_fano* ef_ptr = ef_bitstreams[list_no];
        if (!ef_ptr) continue; 
        compressed_ids_size_in_bytes += ef_ptr->m_low_bits.size() + ef_ptr->m_high_bits.size();
        for (const auto& codes_this_voronoi_cell : codes_all) {
            codes_size_in_bytes += codes_this_voronoi_cell.size() * sizeof(uint8_t);
        }
    }
    compressed_ids_size_in_bytes /= 8;

}

CompressedIDInvertedListsEliasFano::~CompressedIDInvertedListsEliasFano() {
    for (auto ef_ptr : ef_bitstreams) {
        delete ef_ptr;
    }
}

const faiss::idx_t* CompressedIDInvertedListsEliasFano::get_ids(size_t list_no) const  {
    size_t ls = list_size(list_no); 
    if (ls == 0) {
        return nullptr;
    }
    idx_t *the_ids = new idx_t[ls]; 
    const auto& ef = ef_bitstreams[list_no];
    
    /*
    for (size_t i=0; i < ls; i++) {
        the_ids[i] = ef->select(i);
    }*/

    succinct::elias_fano::select_enumerator it(*ef, 0); 
    for (size_t i=0; i < ls; i++) {
        the_ids[i] = it.next();
    }
    
    return the_ids;        
}


faiss::idx_t CompressedIDInvertedListsEliasFano::get_single_id(size_t list_no, size_t offset) const {
    const auto& ef = ef_bitstreams[list_no];
    
    return ef->select(offset);
}

void CompressedIDInvertedListsEliasFano::release_ids(size_t list_no, const idx_t* ids_in) const  {
    delete [] ids_in; 
}

std::vector<std::pair<uint64_t, const uint8_t*>>
CompressedIDInvertedListsEliasFano::canonicalize_order_inplace(
    const uint64_t* ids_data,
    const uint8_t* codes_data,
    size_t ls
) {
    std::vector<std::pair<uint64_t, const uint8_t*>> pairs;
    for (size_t i = 0; i < ls; ++i) {
        pairs.push_back(std::make_pair(ids_data[i], codes_data + i * code_size));
    }

    // Sort the pairs based on the id
    std::sort(pairs.begin(), pairs.end());

    return pairs;
}

/********************************************************************************* 
 *  Wavelet Tree 
 *********************************************************************************/ 


CompressedIDInvertedListsWaveletTree::CompressedIDInvertedListsWaveletTree(const faiss::InvertedLists & il, int wt_type): 
    InvertedListsArrayCodes(il), wt_type(wt_type) {
    assert(wt_type == 0 || wt_type == 1);
    size_t ntotal = il.compute_ntotal();
    sdsl::int_vector<> list_nos(ntotal); 
    codes_all.resize(nlist);  
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t ls = il.list_size(list_no); // number of elements in voronoi cell
        ScopedIds ids_in(&il, list_no);
        ScopedCodes codes(&il, list_no);
        const idx_t* ids_data = ids_in.get();
        idx_t prev_id = -1; 
        for (idx_t i = 0; i < ls; i++) {
            assert(ids_data[i] > prev_id); // assume ordered 
            assert(ids_data[i] < ntotal); 
            list_nos[ids_data[i]] = list_no; 
            prev_id = ids_data[i];
        }
        codes_all[list_no].resize(code_size * ls);
        memcpy(codes_all[list_no].data(), codes.get(), codes_all[list_no].size()); 
    }
    if (wt_type == 0) {
        sdsl::construct_im(wt, list_nos);
        compressed_ids_size_in_bytes = sdsl::size_in_bytes(wt); 
    } else {
        sdsl::construct_im(wt_compressed, list_nos);
        compressed_ids_size_in_bytes = sdsl::size_in_bytes(wt_compressed); 
    }
    
}

idx_t CompressedIDInvertedListsWaveletTree::get_single_id(size_t list_no, size_t offset) const {
    return wt_type == 0 ? wt.select(offset + 1, list_no) : wt_compressed.select(offset + 1, list_no); 
}

const idx_t* CompressedIDInvertedListsWaveletTree::get_ids(size_t list_no) const {
    size_t ls = list_size(list_no); 

    idx_t *the_ids = new idx_t[ls]; 

    // there is probably an iterator for this as well...     
    for (size_t i=0; i < ls; i++) {
        the_ids[i] = get_single_id(list_no, i);
    }
    
    return the_ids;        
}


void CompressedIDInvertedListsWaveletTree::release_ids(size_t list_no, const idx_t* ids_in) const {
    delete [] ids_in; 
}

/********************************************************************************* 
 *  Deferred decoding 
 *********************************************************************************/ 


using idx_t = int64_t; 
using namespace faiss;

void search_IVF_defer_id_decoding(
        const IndexIVF & index,
        idx_t n, 
        const float *x, 
        int k, 
        float *distances,
        idx_t *labels, 
        bool decode_1by1, 
        uint8_t* codes,
        bool include_listno) {
    std::unique_ptr<float []> Dq(new float[n * index.nprobe]); 
    std::unique_ptr<idx_t []> Iq(new idx_t[n * index.nprobe]); 
    
    FAISS_THROW_IF_NOT_MSG(
        index.parallel_mode == 3, 
        "set the parallel mode to 3 otherwise search will be single-threaded");

    index.quantizer->search(
        n, x, index.nprobe, Dq.get(), Iq.get());

    index.search_preassigned(
        n, x, k, Iq.get(), Dq.get(), distances, labels, true); 

    const InvertedLists *invlists = index.invlists; 


    if (codes) {
        // return the codes tables 
        size_t code_size = index.code_size;
        size_t code_size_1 = code_size;
        if (include_listno) {
            code_size_1 += index.coarse_code_size();
        }

#pragma omp parallel for if (n * k > 1000)
        for (idx_t ij = 0; ij < n * k; ij++) {
            idx_t key = labels[ij];
            uint8_t* code1 = codes + ij * code_size_1;

            if (key < 0) {
                // Fill with 0xff
                memset(code1, -1, code_size_1);
            } else {
                int list_no = lo_listno(key);
                int offset = lo_offset(key);
                const uint8_t* cc = invlists->get_single_code(list_no, offset);

                if (include_listno) {
                    index.encode_listno(list_no, code1);
                    code1 += code_size_1 - code_size;
                }
                memcpy(code1, cc, code_size);
            }
        }

    }


    // perform the id translation 
    if (decode_1by1) {
#pragma omp parallel for
        for (size_t i = 0; i < n * k; i++) {
            idx_t & l = labels[i];
            if (l >= 0) {
                l = invlists->get_single_id(lo_listno(l), lo_offset(l)); 
            }
        }
        return; 
    }

    // collect queries with the same listno
    std::unordered_map<idx_t, size_t> counts; 
    size_t nv = 0;
    // maybe we could parallelize this
    for (size_t i = 0; i < n * k; i++) {
        if (labels[i] >= 0) {
            counts[lo_listno(labels[i])] += 1;
            nv++; 
        }
    }
    std::unordered_map<idx_t, size_t> offsets; 
    std::vector<idx_t> lists; // useful only for openmp loops 
    size_t ofs = 0;
    for (auto it : counts) {
        offsets[it.first] = ofs;
        lists.push_back(it.first); 
        ofs += it.second; 
    }
    // maybe we could parallelize this
    std::unique_ptr<idx_t []> resno(new idx_t[nv]); 
    for (size_t i = 0; i < n * k; i++) {
        if (labels[i] >= 0) {
            size_t &ofs = offsets[lo_listno(labels[i])]; 
            resno[ofs++] = i;
        }
    }

    // perform translation per list 
//    for (auto it : offsets) {
//        idx_t list_no = it.first; 
//        size_t end = it.second;
#pragma omp parallel for
    for (size_t i = 0; i < lists.size(); i++) {
        idx_t list_no = lists[i]; 
        size_t end = offsets[list_no]; 
        size_t begin = end - counts[list_no]; 
        InvertedLists::ScopedIds sids(invlists, list_no); 
        const idx_t *ids = sids.get();
        for (size_t j = begin; j < end; j++) {
            idx_t r = resno[j];
            idx_t l = labels[r]; 
            assert(lo_listno(l) == list_no); 
            assert(lo_offset(l) < invlists->list_size(list_no)); 
            /*
            printf("r=%ld, l=%ld, list_no=%ld, lo_offset=%ld id=%ld\n", 
                    r, l, it.first, lo_offset(l), ids[lo_offset(l)]);  */
            labels[r] = ids[lo_offset(l)]; 
        }
    }
}