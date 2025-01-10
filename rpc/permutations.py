import numpy as np
from numpy.typing import NDArray
from rpc.types import uint


def lehmer_encode_inplace(perm: NDArray[uint]) -> None:
    for i in range(perm.shape[0]):
        right = perm[i + 1 :]
        right -= right > perm[i]  # type:ignore


def lehmer_decode_inplace(lehmer_code: NDArray[uint]) -> None:
    n = lehmer_code.shape[0]
    for i in reversed(range(n)):
        right = lehmer_code[i + 1 :]
        right += right >= lehmer_code[i]  # type:ignore


def lehmer_encode(perm: NDArray[uint]) -> NDArray[uint]:
    perm = np.copy(perm)
    for i in range(perm.shape[0]):
        right = perm[i + 1 :]
        right -= right > perm[i]  # type:ignore

    return perm


def lehmer_decode(lehmer_code: NDArray[uint]) -> NDArray[uint]:
    lehmer_code = np.copy(lehmer_code)
    n = lehmer_code.shape[0]
    for i in reversed(range(n)):
        right = lehmer_code[i + 1 :]
        right += right >= lehmer_code[i]  # type:ignore

    return lehmer_code


def compute_applied_permutation(arr: NDArray[uint]) -> NDArray[uint]:
    perm_inv = np.lexsort(arr[:, ::-1].T)
    perm = np.zeros(arr.shape[0], dtype=np.uint32)
    perm[perm_inv] = np.arange(arr.shape[0], dtype=np.uint32)
    return perm  # type: ignore
