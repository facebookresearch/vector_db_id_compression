from lib.fenwick_tree import FenwickTree, Range


def test_Range():
    ftree = FenwickTree(0)
    r = Range(ftree, 10, 10)
    assert r.start == r.freq == 10


def test_FenwickTree():
    ftree = FenwickTree()
    assert ftree.size == 0
    r = ftree.insert_then_forward_lookup(0)
    assert r.start == 0
    assert r.freq == 1

    r = ftree.insert_then_forward_lookup(1)
    assert r.start == 1
    assert r.freq == 1

    r = ftree.insert_then_forward_lookup(0)
    assert r.start == 0
    assert r.freq == 2

    r = ftree.insert_then_forward_lookup(5)
    assert r.start == 3
    assert r.freq == 1
