# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

PYTHON_INCLUDE := $(shell python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())")
CONDA_INCLUDE := ${CONDA_PREFIX}/include
AVX2_LIB := ${CONDA_PREFIX}/lib/libfaiss_avx2.so

SDSL_PATH=/private/home/matthijs/src/test/sdsl-lite/installed/
SDSL_INC=-I${SDSL_PATH}/include 
SDSL_LIB=-L${SDSL_PATH}/lib  -lsdsl -ldivsufsort -ldivsufsort64



COPT= -Wall -fPIC -g -O3 -std=c++17 -fopenmp  -mavx2
# for debugging 
# COPT= -Wall -fPIC -g -std=c++17 -fopenmp

all: test

custom_invlists_wrap.cxx custom_invlists.py: custom_invlists.swig custom_invlists_impl.h
	swig -c++ -python -I${CONDA_INCLUDE} custom_invlists.swig

.cpp.o: 
	g++ ${COPT} -c $< -I${CONDA_INCLUDE} ${SDSL_INC}

codec.o: codec.cpp codec.h

custom_invlists_impl.o: custom_invlists_impl.cpp custom_invlists_impl.h

_custom_invlists.so: custom_invlists_wrap.cxx custom_invlists_impl.o codec.o
	g++ ${COPT} \
	$^ \
	-shared \
	-o $@ \
	-I${PYTHON_INCLUDE} -I${CONDA_INCLUDE} ${SDSL_INC} \
	${AVX2_LIB} ${SDSL_LIB}

test: _custom_invlists.so custom_invlists_wrap.cxx custom_invlists.py
	python -c "import faiss, custom_invlists" && \
	g++ codec.cpp test_codec.cpp -O4 -o test_codec && \
	./test_codec && \
	python -m unittest -v test_compressed_ivfs.py

clean:
	rm -f custom_invlists_wrap.cxx _custom_invlists.so custom_invlists_impl.o codec.o

.PHONY: clean test
