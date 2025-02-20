import typing

import numpy as np

# generic numpy dtypes
DType = typing.TypeVar("DType", bound=np.generic, covariant=True)

# 1D numpy array with typed shape
Vector = np.ndarray[tuple[int], np.dtype[DType]]
