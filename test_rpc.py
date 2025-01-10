import numpy as np
from rpc.codecs import (
    BigUniformCodec,
    BigUniformScalarCodec,
    ROCSortedListCodec,
    SequenceCodec,
    UniformCodec,
    UniformScalarCodec,
    ClusterCodec,
    VectorizedPermutationCodec,
)
from rpc.rans import (
    check_ans_state_equality,
    initialize_ans_state,
    uniform_ans_decode,
    uniform_ans_encode,
)
from sortedcontainers import SortedList


# Test rpc/rans.py
def test_uniform_ans_encode_decode() -> None:
    ans_state = initialize_ans_state((2, 3), randomize=True)
    symbols = np.array([0, 1, 20, 1, 5, 3], dtype=np.uint32).reshape((2, 3))
    precs = np.array([2, 2, 50, 3, 10, 15], dtype=np.uint32).reshape((2, 3))
    new_ans_state = uniform_ans_encode(ans_state, symbols, precs)
    ans_state_dec, symbols_dec = uniform_ans_decode(new_ans_state, precs)

    assert np.all(symbols == symbols_dec)
    assert check_ans_state_equality(ans_state, ans_state_dec)
    assert np.issubdtype(symbols.dtype, np.unsignedinteger)
    assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)

    # make sure the decoded symbol is not a scalar
    assert not np.isscalar(symbols)
    assert not np.isscalar(symbols_dec)


# Test rpc/codecs.py
def test_UniformCodec() -> None:
    precs = np.array([10, 5, 3, 8, 2, 5], dtype=np.uint32).reshape((3, 2))
    ans_state = initialize_ans_state(shape=(1,))
    codec = UniformCodec(precs)
    rng = np.random.default_rng(seed=0)

    seq = rng.integers(0, precs - 1, dtype=np.uint32, size=(100, 3, 2))
    for symbols in reversed(seq):
        ans_state = codec.encode(symbols, ans_state)

    for symbols in seq:
        ans_state, symbols_dec = codec.decode(ans_state)
        assert np.all(symbols == symbols_dec)

    assert np.issubdtype(symbols.dtype, np.unsignedinteger)
    assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)

    # make sure the decoded symbol is not a scalar
    assert not np.isscalar(symbols)
    assert not np.isscalar(symbols_dec)


def test_UniformCodecScalar() -> None:
    for prec in np.arange(1, 100, dtype=np.uint64):
        ans_state = initialize_ans_state(randomize=True)
        symbols = np.random.randint(prec, size=1000, dtype=np.uint32)
        codec = UniformScalarCodec(prec)
        for i in np.flip(np.arange(symbols.shape[0])):
            assert np.isscalar(symbols[i])
            ans_state = codec.encode(symbols[i], ans_state)

        for i in np.arange(symbols.shape[0]):
            ans_state, symbol_dec = codec.decode(ans_state)
            assert np.isscalar(symbol_dec)
            assert np.all(symbols[i] == symbol_dec)


def test_ROCSortedListCodec_with_UniformCodec() -> None:
    for set_size in np.arange(1, 50):
        precs = np.array([set_size], dtype=np.uint32)
        symbol_codec = UniformCodec(precs)
        codec = ROCSortedListCodec(set_size, symbol_codec, copy_input=True)

        sorted_seq = SortedList(
            np.arange(set_size, dtype=np.uint32)[:, None], key=lambda x: x[0]
        )
        ans_state = initialize_ans_state(shape=(1,), randomize=True)
        ans_state = codec.encode(sorted_seq, ans_state)
        ans_state, sorted_seq_dec = codec.decode(ans_state)

        assert np.all(sorted_seq_dec == sorted_seq)
        for symbols in sorted_seq:
            assert np.issubdtype(symbols.dtype, np.unsignedinteger)
            assert not np.isscalar(symbols.dtype)

        for symbols_dec in sorted_seq_dec:
            assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)
            assert not np.isscalar(symbols_dec.dtype)


def test_ROCSortedListCodec_with_UniformScalarCodec() -> None:
    for set_size in [10, 1, 2, 3, 4, 5, 23]:
        prec = np.uint32(set_size)
        symbol_codec = UniformScalarCodec(prec)
        codec = ROCSortedListCodec(set_size, symbol_codec, copy_input=True)

        sorted_seq = SortedList(np.arange(set_size, dtype=np.uint32))
        ans_state = initialize_ans_state(randomize=True)
        ans_state = codec.encode(sorted_seq, ans_state)
        ans_state, sorted_seq_dec = codec.decode(ans_state)

        assert np.all(sorted_seq_dec == sorted_seq)
        for symbols in sorted_seq:
            assert np.issubdtype(symbols.dtype, np.unsignedinteger)
            assert np.isscalar(symbols)

        for symbols_dec in sorted_seq_dec:
            assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)
            assert np.isscalar(symbols_dec)


def test_ROCSortedListCodec_with_BigUniformScalarCodec() -> None:
    for set_size in [10, 1, 2, 3, 4, 5, 23, 64, 128]:
        log_prec = np.uint32(set_size)
        symbol_codec = BigUniformScalarCodec(log_prec)
        codec = ROCSortedListCodec(set_size, symbol_codec, copy_input=True)

        sorted_seq = SortedList(np.arange(set_size, dtype=np.uint32))
        ans_state = initialize_ans_state(randomize=True)
        ans_state = codec.encode(sorted_seq, ans_state)
        ans_state, sorted_seq_dec = codec.decode(ans_state)

        assert np.all(sorted_seq_dec == sorted_seq)
        for symbols in sorted_seq:
            assert np.issubdtype(symbols.dtype, np.unsignedinteger)
            assert np.isscalar(symbols)

        for symbols_dec in sorted_seq_dec:
            assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)
            assert np.isscalar(symbols_dec)


def test_BigUniformCodec() -> None:
    log_prec = np.uint64(64)
    ans_state = initialize_ans_state(shape=(4,))
    codec = BigUniformCodec(log_prec)
    seq = (
        np.linspace(0, (1 << 64) - 1, dtype=np.uint64, num=1000 * 4).reshape((1000, 4)),
    )

    for symbols in reversed(seq):
        ans_state = codec.encode(symbols, ans_state)

    for symbols in seq:
        ans_state, symbols_dec = codec.decode(ans_state)
        assert np.all(symbols == symbols_dec)
        assert np.issubdtype(symbols.dtype, np.unsignedinteger)
        assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)
        assert not np.isscalar(symbols)
        assert not np.isscalar(symbols_dec)


def test_BigUniformScalarCodec() -> None:
    log_prec = np.uint32(64)
    ans_state = initialize_ans_state()
    codec = BigUniformScalarCodec(log_prec)
    seq = np.linspace(1 << 32, (1 << 64) - 1, dtype=np.uint64, num=10)

    for symbol in reversed(seq):
        ans_state = codec.encode(symbol, ans_state)

    for symbol in seq:
        ans_state, symbol_dec = codec.decode(ans_state)

        assert np.isscalar(symbol)
        assert np.isscalar(symbol_dec)
        assert symbol == symbol_dec
        assert np.issubdtype(symbol.dtype, np.unsignedinteger)  # type: ignore
        assert np.issubdtype(symbol_dec.dtype, np.unsignedinteger)  # type: ignore


def test_SequenceCodec() -> None:
    num_elements = 1_000
    precs = np.array([10, 5, 3, 8, 2, 5], dtype=np.uint32).reshape((3, 2))
    symbol_codec = UniformCodec(precs)
    codec = SequenceCodec(num_elements, symbol_codec)

    seq = list(
        (
            np.arange(np.prod(precs.shape) * num_elements, dtype=np.uint32).reshape(
                (num_elements, *precs.shape)
            )
            % precs
        ).astype(np.uint32)
    )

    ans_state = initialize_ans_state(shape=precs.shape)
    ans_state = codec.encode(seq, ans_state)

    ans_state, seq_decoded = codec.decode(ans_state)

    for i in range(num_elements):
        symbols = seq[i]
        symbols_dec = seq_decoded[i]
        assert np.all(symbols == symbols_dec)
        assert np.issubdtype(symbols.dtype, np.unsignedinteger)
        assert np.issubdtype(symbols_dec.dtype, np.unsignedinteger)
        assert not np.isscalar(symbols)
        assert not np.isscalar(symbols_dec)


def test_ClusterCodec() -> None:
    for cluster_size in range(1, 50):
        codec = ClusterCodec(cluster_size)  # type: ignore

        cluster = np.arange(cluster_size)
        ans_state = initialize_ans_state()
        ans_state = codec.encode(cluster, ans_state)
        ans_state, cluster_dec = codec.decode(ans_state)

        assert np.all(cluster == cluster_dec)


def teste_VectorizedPermutationCodec() -> None:
    rng = np.random.default_rng()

    for n in [1_000, 10_000, 25_000]:
        perm = rng.permutation(n)
        for d in [100, 300, 748]:
            ans_state = initialize_ans_state(shape=(d,))
            codec = VectorizedPermutationCodec(n, d)
            ans_state = codec.encode(perm, ans_state)
            _, perm_dec = codec.decode(ans_state)
            assert np.all(perm == perm_dec)
