from typing import Optional

import numpy as np
from craystack.rans import (
    base_message,
    flatten,
    pop_with_finer_prec_uniform,
    push_with_finer_prec_uniform,
)
from numpy.typing import NDArray
from rpc.types import ANSState, uint


def uniform_ans_encode(
    ans_state: ANSState,
    symbols: NDArray[uint],
    precs: NDArray[uint],
) -> ANSState:
    return push_with_finer_prec_uniform(ans_state, symbols, precs, atleast_1d=True)  # type: ignore


def uniform_ans_decode(
    ans_state: ANSState, precs: NDArray[uint]
) -> tuple[ANSState, NDArray[uint]]:
    symbols, pop = pop_with_finer_prec_uniform(ans_state, precs, atleast_1d=True)
    return pop(symbols), symbols


def ans_state_size_in_bytes(ans_state: ANSState) -> int:
    return flatten(ans_state).nbytes


def initialize_ans_state(
    shape: Optional[tuple[int, ...]] = None, randomize: bool = False
) -> ANSState:
    return base_message(shape if shape is not None else 1, randomize)  # type: ignore


def check_ans_state_equality(ans_state: ANSState, other_ans_state: ANSState) -> bool:
    return bool(np.all(flatten(ans_state) == flatten(other_ans_state)))  # type: ignore


def compute_ans_state_size_in_bytes(ans_state: ANSState) -> int:
    return flatten(ans_state).nbytes  # type: ignore
