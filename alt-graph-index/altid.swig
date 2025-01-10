%module altid 

/*
   swig -c++ -python -I$CONDA_PREFIX/include altid.swig && \
       g++ -shared -O3 -g -fPIC  -o _altid.so \
        altid_wrap.cxx altid_impl.cpp \
          -I $CONDA_PREFIX/include \
          -I $( python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())" ) \
          $CONDA_PREFIX/lib/libfaiss_avx2.so \
          && \
          python test_altid.py
 */


// to get uint32_t and friends
%include <stdint.i>

// This means: assume what's declared in these .h files is provided
// by the Faiss module.

#define FAISS_API 

%import(module="faiss") "faiss/MetricType.h"
%import(module="faiss") "faiss/Index.h"
%import(module="faiss") "faiss/impl/NSG.h"
%import(module="faiss") "faiss/IndexNSG.h"

%template(FinalNSGGraph) faiss::nsg::Graph< int32_t >;

%{
#include <faiss/impl/FaissAssert.h>

#include "altid_impl.h"

%} 


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


%include "altid_impl.h"


%inline %{

void NSG_replace_final_graph(faiss::NSG & nsg, faiss::nsg::Graph<int32_t> *graph) {
    nsg.final_graph.reset(graph); 
}


// make an untyped version of the above because there is a big mess-up with the SWIG types 
void search_NSG_and_trace_untyped(
        const faiss::IndexNSG & index, 
        faiss::idx_t n, 
        const float *x, 
        int k, 
        void *labels, 
        float *distances, 
        void * visited_nodes)    
{
   search_NSG_and_trace(index, n, x, k, (faiss::idx_t*)labels, distances, *(std::vector<faiss::idx_t> *)visited_nodes); 
}


%}


%pythoncode %{

import faiss
import numpy as np

def replace_final_graph(self, graph): 
    _altid.NSG_replace_final_graph(self, graph)
    graph.this.disown()

faiss.NSG.replace_final_graph = replace_final_graph

def search_and_trace(self, x, k): 
    n, d = x.shape
    I = np.empty((n, k), dtype='int64')
    D = np.empty((n, k), dtype='float32')
    visited_nodes = faiss.Int64Vector()    
    search_NSG_and_trace_untyped(
        self, n, faiss.swig_ptr(x), k, 
        faiss.swig_ptr(I), faiss.swig_ptr(D), 
        visited_nodes)
    return D, I, faiss.vector_to_array(visited_nodes)

faiss.IndexNSG.search_and_trace = search_and_trace


%}



