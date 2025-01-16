# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

PYTHON_INCLUDE := $(shell python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())")
CONDA_INCLUDE := ${CONDA_PREFIX}/include
AVX2_INCLUDE := ${CONDA_PREFIX}/lib/libfaiss_avx2.so

all: test

altid_wrap.cxx altid.py: altid.swig altid_impl.h 
	swig -c++ -python -I${CONDA_INCLUDE} $<


_altid.so: altid_wrap.cxx altid_impl.cpp ../custom_invlist_cpp/codec.cpp
	g++ -fPIC -O3 -std=c++17 \
	$^ \
	-shared \
	-o $@ \
	-I${PYTHON_INCLUDE} \
	-I${CONDA_INCLUDE} \
	${AVX2_INCLUDE}

test: _altid.so
	python -c "import faiss, altid" && python -m unittest -v test_altid.py

	
clean:
	rm -f altid_wrap.cxx altid.py _altid.so

.PHONY: clean test