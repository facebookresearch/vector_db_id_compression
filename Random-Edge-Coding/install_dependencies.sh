#!/bin/bash

pip install --upgrade pip
pip install --upgrade fastremap joblib tqdm numpy ml_collections
pip install -e .

git clone https://github.com/j-towns/craystack.git craystack_repo
cd craystack_repo
git checkout 992e14d1ecedb1de56eba041bc2d7b72d1084ef3
pip install -e .
cd -
