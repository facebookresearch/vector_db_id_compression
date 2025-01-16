# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# clone repos if needed
git clone git@github.com:ot/succinct.git
git clone git@github.com:dsevero/Random-Edge-Coding.git
pip install -e Random-Edge-Coding


# copy elias fano mod to succinct
cp elias_fano.hpp succinct/