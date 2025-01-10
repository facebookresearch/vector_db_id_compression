%module fenwick_tree

%{
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <stack>
#include <cassert>
#include "../src/fenwick_tree.h"
%}

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <stack>
#include <cassert>
#include "../src/fenwick_tree.h"
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
    Range<T> insert_then_forward_lookup(T symbol);
    Range<T> reverse_lookup_then_remove(int index);
    vector<T> inorder_traversal();
};

template <typename T>
struct Range
{
    FenwickTree<T> *ftree;
    int start;
    int freq;
    Range(FenwickTree<T> *ftree, int start, int freq) : ftree(ftree), start(start), freq(freq) {}
};

%template(FenwickTree) FenwickTree<int>;
%template(Range) Range<int>;