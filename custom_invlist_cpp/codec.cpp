// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "codec.h"

#include <algorithm>
#include <cassert>
#include <set>
#include <cstring>
#include <vector>
#include <stack>
#include <iostream>
#include "../fenwick_tree_cpp/src/fenwick_tree.h"
using namespace std;

constexpr uint64_t rans_l = (uint64_t)1 << 31;

uint64_t pop_with_finer_precision(ANSState &ans_state, uint64_t nmax)
{
    uint64_t head_0 = ans_state.get_head();

    if (head_0 >= nmax * ((rans_l / nmax) << 32))
    {
        ans_state.extend_stack(head_0 & 0xffffffff);
        head_0 >>= 32;
    }

    uint64_t cfs = head_0 % nmax;
    uint64_t head = head_0 / nmax;

    if (head_0 < rans_l)
    {
        uint64_t new_head = ans_state.stack_slice();
        head = new_head | (head << 32);
    }

    ans_state.set_head(head);
    return cfs;
}

void push_with_finer_precision(ANSState &ans_state, size_t symbol, uint64_t nmax)
{
    uint64_t head_0 = ans_state.get_head();

    if (head_0 >= ((rans_l / nmax) << 32))
    {
        ans_state.extend_stack(head_0 & 0xffffffff);
        head_0 >>= 32;
    }

    uint64_t head = head_0 * nmax + symbol;

    if (head < rans_l)
    {
        uint64_t new_head = ans_state.stack_slice();
        head = new_head | (head << 32);
    }

    ans_state.set_head(head);
}

void vrans_push(ANSState &state, uint64_t start, int precision)
{
    uint64_t head = state.get_head();
    // not enough room in the head, push 32 bits to the stack
    if (head >= (rans_l >> precision) << 32)
    {
        state.extend_stack(head & 0xffffffff);
        head >>= 32;
    }
    uint64_t head2 = (head << precision) + start;
    state.set_head(head2);
}

uint64_t vrans_pop(ANSState &state, int precision)
{
    uint64_t head_0 = state.get_head();
    uint64_t cfs = head_0 & (((uint64_t)1 << precision) - 1);
    uint64_t head = head_0 >> precision;
    if (head < rans_l)
    {
        uint64_t new_head = state.stack_slice();
        head = (head << 32) | new_head;
    }
    state.set_head(head);
    return cfs;
}

void codec_push(ANSState &state, uint64_t symbol, int precision)
{
    // encode by 16-bit slices
    for (int lower = 0; lower < 64; lower += 16)
    {
        uint64_t s = (symbol >> lower) & 0xffff;
        int p = precision - lower;
        if (p < 0)
            p = 0;
        if (p > 16)
            p = 16;
        vrans_push(state, s, p);
    }
}

uint64_t codec_pop(ANSState &state, int precision)
{
    uint64_t symbol = 0;
    for (int lower = 48; lower >= 0; lower -= 16)
    {
        int p = precision - lower;
        if (p < 0)
            p = 0;
        if (p > 16)
            p = 16;
        uint64_t s = vrans_pop(state, p);
        symbol = (symbol << 16) | s;
    }
    return symbol;
}

void compress(size_t n, const uint64_t *data, ANSState &state, int precision)
{
    FenwickTree<uint64_t> ftree;
    for (size_t i = 0; i < n; i++)
    {
        ftree.insert_then_forward_lookup(data[i]);
    }

    for (size_t i = 0; i < n; i++)
    {
        uint32_t nmax = n - i;
        size_t index = pop_with_finer_precision(state, nmax);
        auto range = ftree.reverse_lookup_then_remove(index);
        codec_push(state, range.ftree->symbol, precision);
    }
}

void decompress(ANSState &state, size_t n, uint64_t *data, int precision)
{
    FenwickTree<uint64_t> ftree;

    for (size_t i = 0; i < n; i++)
    {
        uint64_t symbol = codec_pop(state, precision);
        auto range = ftree.insert_then_forward_lookup(symbol);
        uint32_t nmax = i + 1;
        push_with_finer_precision(state, range.start, nmax);
        data[n - i - 1] = range.ftree->symbol;
    }
}