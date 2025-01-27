// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FENWICK_TREE_H
#define FENWICK_TREE_H
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <stack>
#include <cassert>
using namespace std;

template <typename T>
struct Range;

template <typename T>
struct FenwickTree
{
    T symbol;
    int size;
    FenwickTree<T> *left;
    FenwickTree<T> *right;

    FenwickTree(T symbol) : symbol(symbol), size(1), left(NULL), right(NULL) {}
    FenwickTree() : symbol(T()), size(0), left(NULL), right(NULL) {}
    ~FenwickTree()
    {
        if (left != NULL)
        {
            delete left;
        }
        if (right != NULL)
        {
            delete right;
        }
    }

    Range<T> insert_then_forward_lookup(T symbol)
    {
        FenwickTree<T> *current = this;
        int size_not_right;
        int freq;
        int start;
        int start_offset = 0;

        if (this->size == 0)
        {
            this->size += 1;
            this->symbol = symbol;
            return Range<T>(this, 0, 1);
        }

        while (true)
        {
            size_not_right = current->size - (current->right != NULL ? current->right->size : 0);
            freq = size_not_right - (current->left != NULL ? current->left->size : 0);
            start = size_not_right - freq;
            current->size += 1;

            if (symbol < current->symbol)
            {
                if (current->left == NULL)
                {
                    current->left = new FenwickTree(symbol);
                    return Range<T>(current->left, start_offset, 1);
                }
                else
                {
                    current = current->left;
                }
            }
            else if (symbol > current->symbol)
            {
                start_offset += size_not_right;
                if (current->right == NULL)
                {
                    current->right = new FenwickTree(symbol);
                    return Range<T>(current->right, start_offset, 1);
                }
                else
                {
                    current = current->right;
                }
            }
            else
            {
                return Range<T>(current, start + start_offset, freq + 1);
            }
        }
    }

    Range<T> reverse_lookup_then_remove(int index)
    {
        FenwickTree<T> *current = this;
        FenwickTree<T> *parent = NULL;
        int size_not_right;
        int freq;
        int start;
        bool went_left;
        int start_offset = 0;

        while (true)
        {
            size_not_right = current->size - (current->right != NULL ? current->right->size : 0);
            freq = size_not_right - (current->left != NULL ? current->left->size : 0);
            start = size_not_right - freq;

            current->size -= 1;
            if (index < start)
            {
                went_left = true;
                parent = current;
                current = current->left;
            }
            else if (index >= start + freq)
            {
                went_left = false;
                parent = current;
                current = current->right;
                index -= size_not_right;
                start_offset += size_not_right;
            }
            else
            {
                if (current->size == 0 && went_left && parent != NULL)
                {
                    parent->left = NULL;
                }
                else if (current->size == 0 && parent != NULL)
                {
                    parent->right = NULL;
                }
                return Range<T>(current, start + start_offset, freq);
            }
        }
    }

    vector<T> inorder_traversal()
    {
        vector<T> elements;
        stack<FenwickTree<T> *> stack;
        FenwickTree<T> *current = this;

        while (current != NULL || stack.empty() == false)
        {
            while (current != NULL)
            {
                stack.push(current);
                current = current->left;
            }
            current = stack.top();
            stack.pop();
            int size_not_right = current->size - (current->right != NULL ? current->right->size : 0);
            int freq = size_not_right - (current->left != NULL ? current->left->size : 0);
            for (int i = 0; i < freq; i++)
            {
                elements.push_back(current->symbol);
            }
            current = current->right;
        }
        return elements;
    }
};

template <typename T>
struct Range
{
    FenwickTree<T> *ftree;
    int start;
    int freq;

    Range(FenwickTree<T> *ftree, int start, int freq) : ftree(ftree), start(start), freq(freq) {}
};

#endif // FENWICK_TREE_H
