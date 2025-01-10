%module custom_invlists; 

/* To compile (Mac OS)

swig -c++ -python -I$CONDA_PREFIX/include custom_invlists.swig &&
g++ -fPIC -O3 -std=c++17 \
custom_invlists_wrap.cxx codec.cpp \
-o _custom_invlists.so \
-I $( python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())" ) \
-I $CONDA_PREFIX/include \
-Wl,-F. -bundle -undefined dynamic_lookup  &&
python -c "import faiss, custom_invlists"


 */


// to get uint32_t and friends
%include <stdint.i>

// Put C++ includes here
%{


#include <faiss/invlists/InvertedLists.h>
#include <faiss/impl/FaissException.h>
#include "codec.h"
#include "custom_invlists_impl.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

%}

#define FAISS_API

%import(module="faiss") "faiss/MetricType.h"
%import(module="faiss") "faiss/invlists/InvertedLists.h"
%import(module="faiss") "faiss/Index.h"
%import(module="faiss") "faiss/IndexIVF.h"



// This is important to release GIL and do Faiss exception handing
%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(faiss::FaissException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc & ba) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}

%ignore CompressedIDInvertedListsWaveletTree::wt;

%include "custom_invlists_impl.h"

%inline %{

// untyped is a work-around for a type mismatch, should be fixed when 
// https://www.internalfb.com/diff/D63991471 lands
void search_IVF_defer_id_decoding_untyped(
        const faiss::IndexIVF & index,
        faiss::idx_t n, 
        const float *x, 
        int k, 
        float *distances, 
        void *labels, 
        bool decode_1by1, 
        void *codes, 
        bool include_listno) {

    search_IVF_defer_id_decoding(
        index, n, x, k, distances, (faiss::idx_t*)labels, decode_1by1, 
        (uint8_t*)codes, include_listno); 
}


%}

%pythoncode %{

import numpy as np
import faiss


def search_IVF_defer_id_decoding(self, x, k, decode_1by1=False, return_codes=0): 
    """
    decode_1by1 = use get_single_id instead of get_ids (if random access is efficient)
    return_codes = 0: no, 1: yes, without listnos, 2: yes, with listnos 
    """

    x = np.ascontiguousarray(x, dtype="float32")
    n, d = x.shape
    assert d == self.d
    D = np.zeros((n, k), dtype="float32")
    I = np.zeros((n, k), dtype="int64")

    if return_codes: 
        code_size_1 = self.code_size
        if return_codes == 2: 
            code_size_1 += self.coarse_code_size()
        codes = np.zeros((n, k, code_size_1), dtype="uint8")
        codes_ptr = faiss.swig_ptr(codes)
    else: 
        codes_ptr = None

    search_IVF_defer_id_decoding_untyped(
        self, n, faiss.swig_ptr(x), k, 
        faiss.swig_ptr(D), faiss.swig_ptr(I), 
        decode_1by1, codes_ptr, return_codes == 2
    )
    
    if return_codes: 
        return D, I, codes 
    else: 
        return D, I            

faiss.IndexIVF.search_defer_id_decoding = search_IVF_defer_id_decoding


%}