# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

PYTHON_INCLUDE := $(shell python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())")
CONDA_INCLUDE := ${CONDA_PREFIX}/include

all: test

lib/fenwick_tree_wrap.cxx: dirs
	swig -c++ -python -outdir lib -o lib/fenwick_tree_wrap.cxx src/fenwick_tree.i

lib/fenwick_tree.o: 
	g++ -std=c++17 -O2 -fPIC -c src/fenwick_tree.cpp -o lib/fenwick_tree.o

lib/fenwick_tree_wrap.o: lib/fenwick_tree_wrap.cxx
	g++ -std=c++17 -O2 -fPIC -c lib/fenwick_tree_wrap.cxx -I${PYTHON_INCLUDE} -o lib/fenwick_tree_wrap.o

lib/_fenwick_tree.so: lib/fenwick_tree.o lib/fenwick_tree_wrap.o
	g++ -std=c++17 -shared lib/fenwick_tree.o lib/fenwick_tree_wrap.o -o lib/_fenwick_tree.so

dirs:
	mkdir -p src bin lib 

bin/test_fenwick_tree: dirs
	g++ tests/test_fenwick_tree.cpp -o bin/test_fenwick_tree

test: bin/test_fenwick_tree lib/_fenwick_tree.so
	./bin/test_fenwick_tree
	python -B -m pytest tests/test_FenwickTree.py -v

clean:
	rm -rf bin/* lib/* tests/__pycache__

.PHONY: clean test