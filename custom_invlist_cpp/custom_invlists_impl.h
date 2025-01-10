

#include <faiss/invlists/InvertedLists.h>
#include <faiss/IndexIVF.h>

#include "codec.h"
#include "../succinct/elias_fano.hpp"

#include <sdsl/wavelet_trees.hpp>
#include <faiss/utils/hamming.h>

uint64_t BitstringReader_get_bits(const faiss::BitstringReader &bs, size_t i, int nbit);

/// all the invlist impelmentations just store the codes in an array, so put this in common
struct InvertedListsArrayCodes: faiss::ReadOnlyInvertedLists {
    using idx_t = faiss::idx_t; 

    std::vector<std::vector<uint8_t>> codes_all; 

    size_t list_size(size_t list_no) const override;

    const uint8_t* get_codes(size_t list_no) const override;

    InvertedListsArrayCodes(const faiss::InvertedLists & il); 

};


/// Packed bits: just allocate the necessary bits to store ids from 0 to notal-1
struct CompressedIDInvertedListsPackedBits: InvertedListsArrayCodes {
    using idx_t = faiss::idx_t; 

    /// bits per identifier 
    int bits = 0;
    std::vector<std::vector<uint8_t>> ids_all; 

    size_t compressed_ids_size_in_bytes = 0;

    CompressedIDInvertedListsPackedBits(const faiss::InvertedLists & il);

    const idx_t* get_ids(size_t list_no) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override; 

    void release_ids(size_t list_no, const idx_t* ids_in) const override; 
};


struct CompressedIDInvertedListsFenwickTree: InvertedListsArrayCodes {
    using idx_t = faiss::idx_t; 

    std::vector<ANSState> ans_states;    
    size_t compressed_ids_size_in_bytes = 0;
    size_t codes_size_in_bytes = 0;
    std::vector<uint64_t> id_symbol_precision;
    size_t overhead_in_bytes = 0;

    CompressedIDInvertedListsFenwickTree(const faiss::InvertedLists & il);

    const idx_t* get_ids(size_t list_no) const override;

    void release_ids(size_t list_no, const idx_t* ids_in) const override; 
};

struct CompressedIDInvertedListsEliasFano: InvertedListsArrayCodes {
    using idx_t = faiss::idx_t; 
    size_t overhead_in_bytes = 0;

    std::vector<succinct::elias_fano*> ef_bitstreams;

    size_t compressed_ids_size_in_bytes = 0;
    size_t codes_size_in_bytes = 0;

    CompressedIDInvertedListsEliasFano(const faiss::InvertedLists & il);

    ~CompressedIDInvertedListsEliasFano();

    const idx_t* get_ids(size_t list_no) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override; 

    void release_ids(size_t list_no, const idx_t* ids_in) const override;

    std::vector<std::pair<uint64_t, const uint8_t*>>
    canonicalize_order_inplace(
        const uint64_t* ids_data,
        const uint8_t* codes_data,
        size_t ls
    );

};

struct CompressedIDInvertedListsWaveletTree: InvertedListsArrayCodes {
    using idx_t = faiss::idx_t; 

    // the tree structure 
    sdsl::wt_int<sdsl::bit_vector> wt;
    sdsl::wt_int<sdsl::rrr_vector<63>> wt_compressed;

    size_t compressed_ids_size_in_bytes = 0;

    int wt_type = 0; 

    // vector_type == 0: use wt 
    // vector_type == 1: use wt_compressed
    CompressedIDInvertedListsWaveletTree(const faiss::InvertedLists & il, int wt_type=0);

    // ~CompressedIDInvertedListsWaveletTree(); 

    const idx_t* get_ids(size_t list_no) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override; 

    void release_ids(size_t list_no, const idx_t* ids_in) const override;


}; 

/* Search an IVF. 
 * Do not collect the result IDs immediately, but instead collect (invlist, offset) pairs
 * and do the conversion to IDs when the search finished -- avoids decompressing invlists 
 * without necessity */
void search_IVF_defer_id_decoding(
        const faiss::IndexIVF & index,
        faiss::idx_t n, 
        const float *x, 
        int k, 
        float *distances,
        faiss::idx_t *labels, 
        bool decode_1by1 = false, 
        uint8_t* codes = nullptr,
        bool include_listno = false);

