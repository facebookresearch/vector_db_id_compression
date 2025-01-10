from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

# Type for the nested tuple ANS state in craystack.
type ANSState = tuple[NDArray[np.uint64], "ANSState" | tuple[()]]

# Type allowing any unsigned integer numpy type.
uint = TypeVar("uint", bound=np.unsignedinteger[Any])
