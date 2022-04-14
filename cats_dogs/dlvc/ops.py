from typing import List, Callable

import numpy as np

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    # TODO implement (see above for guidance).
    def cast(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)

    return cast

def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    # TODO implement (see above for guidance).
    return np.ravel

def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def add_value(sample: np.ndarray) -> np.ndarray:
        return sample + val
    return add_value

def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def multiply_value(sample: np.ndarray) -> np.ndarray:
        return sample * val
    return multiply_value

def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''
    def flipped(sample: np.ndarray) -> np.ndarray:
        return sample.transpose(2, 0, 1)
    return flipped


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''
    def flipped(sample: np.ndarray) -> np.ndarray:
        if np.random.choice([True, False]):
            return np.flip(sample, (0,2))
        else:
            return sample
    return flipped

def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # TODO implement
    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html will be helpful
    def cropped(sample: np.ndarray) -> np.ndarray:
        if pad > 0:
            sample = np.pad(sample, pad, pad_mode)
        h, w, _ = sample.shape
        if sz > h or sz > w:
            raise ValueError("square size bigger than one of dimensions")
        top, left = np.random.randint(0, w-sz), np.random.randint(0, h-sz)
        return sample[top:top+sz, left:left+sz, :]
    return cropped