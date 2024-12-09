"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1
def mul(x: float, y: float) -> float:
    """Multiplication function $f(x, y) = x * y$"""
    return x * y


def id(x: float) -> float:  # noqa: D103
    """Identity function $f(x) = x$"""
    return x


def add(x: float, y: float) -> float:  # noqa: D103
    """Addition function $f(x, y) = x + y$"""
    return x + y


def neg(x: float) -> float:  # noqa: D103
    """Negation function $f(x) = -x$"""
    return -x


def lt(x: float, y: float) -> float:  # noqa: D103
    """Less than function $f(x, y) = 1$ if $x < y$ else $0$"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:  # noqa: D103
    """Equality function $f(x, y) = 1$ if $x == y$ else $0$"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:  # noqa: D103
    r"""Maximum function $f(x, y) = \max(x, y)$"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:  # noqa: D103
    """Close function $f(x, y) = 1$ if $|x - y| < 1e-2$ else $0$"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:  # noqa: D103
    r"""Sigmoid function $f(x) = \frac{1.0}{(1.0 + e^{-x})}$ if $x >= 0$ else $\frac{e^x}{(1.0 + e^{x})}$"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:  # noqa: D103
    """ReLU function $f(x) = x$ if $x > 0$ else $0$"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:  # noqa: D103
    r"""Logarithm function $f(x) = \log(x + 1e-6)$"""
    return math.log(x + EPS)


def exp(x: float) -> float:  # noqa: D103
    """Exponential function $f(x) = e^x$"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:  # noqa: D103
    r"""Logarithm back function $f(x, d) = \frac{d}{x + 1e-6}$"""
    return d / (x + EPS)


def inv(x: float) -> float:  # noqa: D103
    """Inverse function $f(x) = 1.0 / x$"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:  # noqa: D103
    r"""Inverse back function $f(x, d) = -\frac{d}{x^2}$"""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:  # noqa: D103
    """ReLU back function $f(x, d) = d$ if $x > 0$ else $0$"""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
# Core Functions:
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:  # noqa: D103
    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(  # noqa: D103
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:  # noqa: D103
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(  # noqa: D103
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:  # noqa: D103
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:  # type: ignore  # noqa: D103
    return zipWith(add)(ls1, ls2)


def negList(ls: Iterable[float]) -> Iterable[float]:  # type: ignore  # noqa: D103
    return map(neg)(ls)


def sum(ls: Iterable[float]) -> float:  # noqa: D103 # type: ignore
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:  # noqa: D103
    return reduce(mul, 1.0)(ls)
