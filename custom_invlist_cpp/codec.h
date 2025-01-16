// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <vector>
#include <random>

#pragma once 

struct ANSState {
    uint64_t head = uint64_t(1) << 31; 
    std::vector<uint32_t> stack;
    std::mt19937 mt;

    ANSState(): mt(1234) {}

    void extend_stack(uint32_t x) {
        stack.push_back(x); 
    }       

    void set_head(uint64_t x) {
        head = x; 
    }

    uint64_t get_head() const {
        return head; 
    }

    uint32_t stack_slice() {
        if (!stack.empty()) {
            uint32_t ret = stack.back(); 
            stack.pop_back(); 
            return ret;
        } else {
            return mt(); 
        }
    }

    size_t size() {
        return 8 + stack.size() * sizeof(uint32_t);
    }
};

void compress(size_t n, const uint64_t *data, ANSState & state, int precision);
void decompress(ANSState & state, size_t n, uint64_t *data, int precision);
uint64_t pop_with_finer_precision(ANSState &ans_state, uint64_t nmax);
void codec_push(ANSState &state, uint64_t symbol, int precision);
void push_with_finer_precision(ANSState &ans_state, size_t symbol, uint64_t nmax);
uint64_t codec_pop(ANSState &state, int precision);