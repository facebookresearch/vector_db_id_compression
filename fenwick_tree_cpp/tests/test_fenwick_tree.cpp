#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <stack>
#include <cassert>
#include "../src/fenwick_tree.h"
using namespace std;

void test_FenwickTree_1()
{
    FenwickTree<char> ftree;

    // INSERT
    // auto range = insert_then_forward_lookup(&ftree, 'b');
    auto range = ftree.insert_then_forward_lookup('b');
    assert(range.ftree->symbol == 'b');
    assert(range.start == 0);
    assert(range.freq == 1);
    auto expected = vector<char>{'b'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('a');
    assert(range.ftree->symbol == 'a');
    assert(range.start == 0);
    assert(range.freq == 1);
    expected = vector<char>{'a', 'b'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('b');
    assert(range.ftree->symbol == 'b');
    assert(range.start == 1);
    assert(range.freq == 2);
    expected = vector<char>{'a', 'b', 'b'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('d');
    assert(range.ftree->symbol == 'd');
    assert(range.start == 3);
    assert(range.freq == 1);
    expected = vector<char>{'a', 'b', 'b', 'd'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('c');
    assert(range.ftree->symbol == 'c');
    assert(range.start == 3);
    assert(range.freq == 1);
    expected = vector<char>{'a', 'b', 'b', 'c', 'd'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('e');
    assert(range.ftree->symbol == 'e');
    assert(range.start == 5);
    assert(range.freq == 1);
    expected = vector<char>{'a', 'b', 'b', 'c', 'd', 'e'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('c');
    assert(range.ftree->symbol == 'c');
    assert(range.start == 3);
    assert(range.freq == 2);
    expected = vector<char>{'a', 'b', 'b', 'c', 'c', 'd', 'e'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup('c');
    assert(range.ftree->symbol == 'c');
    assert(range.start == 3);
    assert(range.freq == 3);
    expected = vector<char>{'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e'};
    assert(ftree.inorder_traversal() == expected);

    // REMOVE
    range = ftree.reverse_lookup_then_remove(6);
    assert(range.ftree->symbol == 'd');
    assert(range.start == 6);
    assert(range.freq == 1);
    expected = vector<char>{'a', 'b', 'b', 'c', 'c', 'c', 'e'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(1);
    assert(range.ftree->symbol == 'b');
    assert(range.start == 1);
    assert(range.freq == 2);
    expected = vector<char>{'a', 'b', 'c', 'c', 'c', 'e'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(3);
    assert(range.ftree->symbol == 'c');
    assert(range.start == 2);
    assert(range.freq == 3);
    expected = vector<char>{'a', 'b', 'c', 'c', 'e'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(4);
    assert(range.ftree->symbol == 'e');
    assert(range.start == 4);
    assert(range.freq == 1);
    expected = vector<char>{'a', 'b', 'c', 'c'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(0);
    assert(range.ftree->symbol == 'a');
    assert(range.start == 0);
    assert(range.freq == 1);
    expected = vector<char>{'b', 'c', 'c'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(1);
    assert(range.ftree->symbol == 'c');
    assert(range.start == 1);
    assert(range.freq == 2);
    expected = vector<char>{'b', 'c'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(0);
    assert(range.ftree->symbol == 'b');
    assert(range.start == 0);
    assert(range.freq == 1);
    expected = vector<char>{'c'};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.reverse_lookup_then_remove(0);
    assert(range.ftree->symbol == 'c');
    assert(range.start == 0);
    assert(range.freq == 1);
    expected = vector<char>{};
    assert(ftree.inorder_traversal() == expected);
}

void test_FenwickTree_2()
{
    FenwickTree<uint64_t> ftree;

    // INSERT
    auto range = ftree.insert_then_forward_lookup((uint64_t)83);
    assert(range.ftree->symbol == 83);
    assert(range.start == 0);
    assert(range.freq == 1);
    auto expected = vector<uint64_t>{83};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup((uint64_t)77);
    assert(range.ftree->symbol == 77);
    assert(range.start == 0);
    assert(range.freq == 1);
    expected = vector<uint64_t>{77, 83};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup((uint64_t)15);
    assert(range.ftree->symbol == 15);
    assert(range.start == 0);
    assert(range.freq == 1);
    expected = vector<uint64_t>{15, 77, 83};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup((uint64_t)86);
    assert(range.ftree->symbol == 86);
    assert(range.start == 3);
    assert(range.freq == 1);
    expected = vector<uint64_t>{15, 77, 83, 86};
    assert(ftree.inorder_traversal() == expected);

    range = ftree.insert_then_forward_lookup((uint64_t)93);
    assert(range.ftree->symbol == 93);
    assert(range.start == 4);
    assert(range.freq == 1);
    expected = vector<uint64_t>{15, 77, 83, 86, 93};
    assert(ftree.inorder_traversal() == expected);

    // REMOVE
    range = ftree.reverse_lookup_then_remove(3);
    assert(range.ftree->symbol == 86);
    assert(range.start == 3);
    assert(range.freq == 1);
    expected = vector<uint64_t>{15, 77, 83, 93};
    assert(ftree.inorder_traversal() == expected);
}

int main()
{
    cout << "Running tests ..." << endl;
    test_FenwickTree_1();
    test_FenwickTree_2();
    cout << "All tests passed!" << endl;
    return 0;
}