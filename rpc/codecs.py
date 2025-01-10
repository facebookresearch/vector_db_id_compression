import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, Sequence, TypeVar

import numpy as np
from craystack import BigUniform
from numpy.typing import NDArray
from rpc.permutations import compute_applied_permutation, lehmer_decode, lehmer_encode
from rpc.rans import uniform_ans_decode, uniform_ans_encode
from rpc.types import ANSState, uint
from sortedcontainers import SortedList

T = TypeVar("T")


class Codec[SymbolType](ABC):
    @abstractmethod
    def encode(self, symbols: SymbolType, ans_state: ANSState) -> ANSState:
        pass

    @abstractmethod
    def decode(self, ans_state: ANSState) -> tuple[ANSState, SymbolType]:
        pass


@dataclass
class UniformCodec(Codec[NDArray[uint]]):
    precs: NDArray[uint]

    def encode(self, symbols: NDArray[uint], ans_state: ANSState) -> ANSState:
        return uniform_ans_encode(ans_state, symbols, self.precs)

    def decode(self, ans_state: ANSState) -> tuple[ANSState, NDArray[uint]]:
        return uniform_ans_decode(ans_state, self.precs)


@dataclass
class UniformScalarCodec(Codec[uint]):
    prec: uint

    def __post_init__(self) -> None:
        self.precs = np.array(self.prec)
        self.vectorized_codec = UniformCodec(self.precs)

    def encode(self, symbol: uint, ans_state: ANSState) -> ANSState:
        return self.vectorized_codec.encode(np.array([symbol]), ans_state)

    def decode(self, ans_state: ANSState) -> tuple[ANSState, uint]:
        ans_state, symbols = self.vectorized_codec.decode(ans_state)
        return ans_state, symbols[0]


@dataclass
class BigUniformCodec(Codec[NDArray[uint]]):
    """Currently only handles a single precision value for all streams."""

    log_prec: uint

    def __post_init__(self) -> None:
        self.codec = BigUniform(self.log_prec)  # type: ignore

    def encode(self, symbols: NDArray[uint], ans_state: ANSState) -> ANSState:
        (ans_state,) = self.codec.push(ans_state, symbols)
        return ans_state

    def decode(self, ans_state: ANSState) -> tuple[ANSState, NDArray[uint]]:
        ans_state, symbols = self.codec.pop(ans_state)
        return ans_state, symbols


@dataclass
class BigUniformScalarCodec(Codec[uint]):
    log_prec: uint

    def __post_init__(self) -> None:
        self.vectorized_codec = BigUniformCodec(self.log_prec)

    def encode(self, symbol: uint, ans_state: ANSState) -> ANSState:
        return self.vectorized_codec.encode(np.array([symbol]), ans_state)

    def decode(self, ans_state: ANSState) -> tuple[ANSState, uint]:
        ans_state, symbols = self.vectorized_codec.decode(ans_state)
        return ans_state, symbols[0]


@dataclass
class ROCSortedListCodec(Codec[SortedList[T]]):
    set_size: int
    symbol_codec: Codec[T]
    key: Optional[Callable[[T], Any]] = None
    copy_input: bool = True

    def encode(self, sorted_seq: SortedList[T], ans_state: ANSState) -> ANSState:
        # This can be removed when going to production.
        if self.copy_input:
            sorted_seq = deepcopy(sorted_seq)

        for i in range(self.set_size):
            # Sample/Decode, without replacement, an index using ANS.
            # Initialize a uniform codec for the indices.
            prec = np.uint32(self.set_size - i)
            ans_state, index = UniformScalarCodec(prec).decode(ans_state)

            # `index` is NDArray[uint], need to cast to int to pick the element.
            symbol: T = sorted_seq.pop(int(index))

            # Encode the element into the ans state.
            ans_state = self.symbol_codec.encode(symbol, ans_state)

        return ans_state

    def decode(self, ans_state: ANSState) -> tuple[ANSState, SortedList[T]]:
        # Initialize an empty SortedList of elements of type T.
        sorted_seq: SortedList[T] = SortedList([], key=self.key)

        for i in range(self.set_size):
            # Decode an element from the stack.
            ans_state, symbol = self.symbol_codec.decode(ans_state)

            # Add it to the sorted list and recover `index`.
            sorted_seq.add(symbol)
            index = np.uint32(sorted_seq.index(symbol))

            # Encode `index` back into the state to reverse sampling.
            prec = np.uint32(i + 1)
            ans_state = UniformScalarCodec(prec).encode(index, ans_state)

        return ans_state, sorted_seq


@dataclass
class ROCSortedListCodecTimed:
    set_size: int
    symbol_codec: Codec[T]
    key: Optional[Callable[[T], Any]] = None
    copy_input: bool = True

    def encode(self, sorted_seq: SortedList[T], ans_state: ANSState) -> ANSState:
        # This can be removed when going to production.
        if self.copy_input:
            sorted_seq = deepcopy(sorted_seq)

        for i in range(self.set_size):
            # Sample/Decode, without replacement, an index using ANS.
            # Initialize a uniform codec for the indices.
            prec = np.uint32(self.set_size - i)
            ans_state, index = UniformScalarCodec(prec).decode(ans_state)

            # `index` is NDArray[uint], need to cast to int to pick the element.
            symbol: T = sorted_seq.pop(int(index))

            # Encode the element into the ans state.
            ans_state = self.symbol_codec.encode(symbol, ans_state)

        return ans_state

    def decode(
        self, ans_state: ANSState
    ) -> tuple[ANSState, SortedList[T], float, float]:
        # Initialize an empty SortedList of elements of type T.
        sorted_seq: SortedList[T] = SortedList([], key=self.key)

        dt_symbols = 0.0
        dt_perm = 0.0
        for i in range(self.set_size):
            t0 = time.perf_counter()
            # Decode an element from the stack.
            ans_state, symbol = self.symbol_codec.decode(ans_state)
            t1 = time.perf_counter()

            # Add it to the sorted list and recover `index`.
            sorted_seq.add(symbol)
            index = np.uint32(sorted_seq.index(symbol))

            # Encode `index` back into the state to reverse sampling.
            prec = np.uint32(i + 1)
            ans_state = UniformScalarCodec(prec).encode(index, ans_state)
            t2 = time.perf_counter()

            dt_symbols += t1 - t0
            dt_perm += t2 - t1

        return ans_state, sorted_seq, dt_symbols, dt_perm


@dataclass
class SequenceCodec(Codec[Sequence[T]]):
    num_elements: int
    symbol_codec: Codec[T]

    def encode(self, seq: Sequence[T], ans_state: ANSState) -> ANSState:
        for symbol in reversed(seq):
            ans_state = self.symbol_codec.encode(symbol, ans_state)
        return ans_state

    def decode(self, ans_state: ANSState) -> tuple[ANSState, Sequence[T]]:
        seq = []
        for _ in range(self.num_elements):
            ans_state, symbol = self.symbol_codec.decode(ans_state)
            seq.append(symbol)
        return ans_state, seq

    def decode_streaming(
        self, ans_state: ANSState
    ) -> Generator[tuple[ANSState, T], None, None]:
        for _ in range(self.num_elements):
            yield self.symbol_codec.decode(ans_state)


@dataclass
class ClusterCodec(Codec[NDArray[uint]]):
    cluster_size: int

    def __post_init__(self) -> None:
        self.roc_codec = ROCSortedListCodec(
            set_size=self.cluster_size,
            symbol_codec=BigUniformScalarCodec(log_prec=np.uint32(64)),
            copy_input=False,
        )

    def encode(self, cluster_ids: NDArray[uint], ans_state: ANSState) -> ANSState:
        sorted_seq = SortedList(cluster_ids)
        ans_state = self.roc_codec.encode(sorted_seq, ans_state)
        return ans_state

    def decode(self, ans_state: ANSState) -> tuple[ANSState, NDArray[uint]]:
        ans_state, sorted_seq = self.roc_codec.decode(ans_state)
        return ans_state, np.array(sorted_seq)


@dataclass
class VectorizedPermutationCodec(Codec[NDArray[uint]]):
    permutation_size: int
    ans_state_size: int

    def __post_init__(self) -> None:
        precs_all = np.arange(self.permutation_size, dtype=np.uint32)[::-1]
        precs_all = self._maybe_pad(precs_all) + 1  # type: ignore
        num_splits = len(precs_all) // self.ans_state_size
        self.uniform_codecs = [
            UniformCodec(precs) for precs in np.array_split(precs_all, num_splits)
        ]
        self.num_sub_vectors = len(self.uniform_codecs)

    def _maybe_pad(self, x: NDArray[uint]) -> NDArray[uint]:
        if self.permutation_size % self.ans_state_size != 0:
            pad_width = (
                self.ans_state_size - self.permutation_size % self.ans_state_size
            )
            x = np.pad(x, pad_width=(0, pad_width))
        return x

    def encode(self, perm: NDArray[uint], ans_state: ANSState) -> ANSState:
        d = self.ans_state_size

        lehmer_code = lehmer_encode(perm)
        lehmer_code = self._maybe_pad(lehmer_code)

        for i, unif_codec in enumerate(self.uniform_codecs):
            symbols = lehmer_code[i * d : (i + 1) * d]
            ans_state = unif_codec.encode(symbols, ans_state)  # type: ignore
        return ans_state

    def decode(self, ans_state: ANSState) -> tuple[ANSState, NDArray[uint]]:
        n = self.permutation_size
        k = self.num_sub_vectors
        d = self.ans_state_size

        lehmer_code = self._maybe_pad(np.zeros(self.permutation_size, dtype=np.uint32))
        for i, unif_codec in enumerate(reversed(self.uniform_codecs)):
            ans_state, lehmer_code[(k - i - 1) * d : (k - i) * d] = unif_codec.decode(
                ans_state
            )
        perm = lehmer_decode(lehmer_code[:n])
        return ans_state, perm


@dataclass
class ROCOneShotArrayCodec(Codec[NDArray[uint]]):
    array_shape: tuple[int, int]
    row_codec: Codec[NDArray[uint]]

    def __post_init__(self) -> None:
        self.permutation_codec = VectorizedPermutationCodec(
            *self.array_shape
        )  # type:ignore

    def encode(self, sorted_array: NDArray[uint], ans_state: ANSState) -> ANSState:
        ans_state, perm = self.permutation_codec.decode(ans_state)

        for i in perm:
            ans_state = self.row_codec.encode(sorted_array[i], ans_state)

        return ans_state

    def decode(self, ans_state: ANSState) -> tuple[ANSState, NDArray[uint]]:
        n, d = self.array_shape
        array_decoded = np.zeros((n, d), dtype=np.uint32)
        for i in reversed(range(n)):
            ans_state, array_decoded[i] = self.row_codec.decode(ans_state)

        # array_decoded is now ready to be used, up to here all computations are O(n)
        # the rest can be done in parallel by another thread
        # inverting the permutation can be O(n*log(n))

        perm = compute_applied_permutation(array_decoded)
        ans_state = self.permutation_codec.encode(perm, ans_state)

        return ans_state, array_decoded


@dataclass
class ROCOneShotArrayCodecTimed:
    array_shape: tuple[int, int]
    row_codec: Codec[NDArray[uint]]

    def __post_init__(self) -> None:
        self.permutation_codec = VectorizedPermutationCodec(
            *self.array_shape
        )  # type:ignore

    def encode(self, sorted_array: NDArray[uint], ans_state: ANSState) -> ANSState:
        ans_state, perm = self.permutation_codec.decode(ans_state)

        for i in perm:
            ans_state = self.row_codec.encode(sorted_array[i], ans_state)

        return ans_state

    def decode(
        self, ans_state: ANSState
    ) -> tuple[ANSState, NDArray[uint], float, float]:
        t0 = time.process_time()
        n, d = self.array_shape
        array_decoded = np.zeros((n, d), dtype=np.uint32)
        for i in reversed(range(n)):
            ans_state, array_decoded[i] = self.row_codec.decode(ans_state)

        t1 = time.process_time()

        # array_decoded is now ready to be used, up to here all computations are O(n)
        # the rest can be done in parallel by another thread
        # inverting the permutation can be O(n*log(n))

        perm = compute_applied_permutation(array_decoded)
        ans_state = self.permutation_codec.encode(perm, ans_state)

        t2 = time.process_time()

        return ans_state, array_decoded, t1 - t0, t2 - t1
