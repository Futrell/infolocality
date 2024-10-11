import csv
import random
import operator
import itertools
import functools
from typing import *

import numpy as np
import pandas as pd

DELIMITER = '#'
EPSILON = 10 ** -8

flat = itertools.chain.from_iterable

T = TypeVar("T", bound=Any)

def flatmap(f, *xss):
    return flat(map(f, *xss))

def shuffled(xs: Iterable[T]) -> List[T]:
    xs = list(xs)
    random.shuffle(xs)
    return xs

def first(xs: Iterable[T]) -> T:
    return next(iter(xs))

def is_monotonically_increasing(xs: Sequence) -> bool:
    return (np.diff(xs) > -EPSILON).all()

def all_same(xs: Iterable) -> bool:
    x, *rest = xs
    return all(r == x for r in rest)

def the_only(xs: Iterable[T]) -> T:
    """ Return the single value of a one-element iterable """
    x, = xs
    return x

def the_unique(xs: Iterable[T]) -> T:
    """ Return the unique value of an iterable of equal values. 
    Example: the_unique([3,3,3,3]) -> 3
    """
    first, *rest = xs
    assert all(r == first for r in rest)
    return first

def cartesian_indices(V: int, k: int) -> Iterator[Tuple[int]]:
    return itertools.product(*[range(V)]*k)

def cartesian_power(xs: T, k: int) -> Iterator[Tuple[T]]:
    xs = list(xs)
    return itertools.product(*[xs]*k)

def cartesian_distinct_indices(V: int, k: int) -> Iterator[Tuple[int]]:
    for sequence in cartesian_indices(V, k):
        yield tuple(i*V + x for i, x in enumerate(sequence))

def cartesian_distinct_forms(V: int, k: int) -> Iterator[str]:
    numerals = cartesian_indices(V, k)
    for group in numerals:
        yield "".join(int_to_char(x+i*V) for i, x in enumerate(group))

def cartesian_forms(V: int, k: int) -> Iterable[str]:
    numerals = cartesian_indices(V, k)
    for group in numerals:
        yield "".join(map(int_to_char, group))

def star_upto(V: int, k: int) -> Iterable[Tuple[int]]:
    for i in range(1, k+1):
        yield from utils.cartesian_indices(V, i)

def int_to_char(x: int, offset: int=65) -> str:
    return chr(offset + x)

def ints_to_str(ints: Iterable[int], offset: int=65) -> str:
    return "".join(map(int_to_char, ints))

def write_dicts(file, lines):
    lines_iter = iter(lines)
    first_line = next(lines_iter)
    writer = csv.DictWriter(file, first_line.keys())
    writer.writeheader()
    writer.writerow(first_line)
    for line in lines_iter:
        writer.writerow(line)        

def write_dfs(file, dfs):
    def gen():
        for df in dfs:
            for _, row in df.iterrows():
                yield dict(row)
    write_dicts(file, gen())


class Delimiter:
    def parts(self, x: T) -> Iterable[Union[str, T]]:
        raise NotImplementedError

    def delimit(self, x: Sequence) -> Sequence:
        return restorer(x)(flat(self.parts(x)))

    def delimit_string(self, x: str) -> str:
        return "".join(self.parts(x))

    def delimit_sequence(self, x: Sequence[T]) -> Tuple[Union[str, T]]:
        return tuple(flat(self.parts(x)))

    def delimit_array(self, x: pd.Series) -> pd.Series:
        return functools.reduce(operator.add, self.parts(x))

class LeftDelimiter(Delimiter):
    def parts(self, x: T) -> List[Union[str, T]]:
        return [DELIMITER, x]

class RightDelimiter(Delimiter):
    def parts(self, x: T) -> List[Union[str, T]]:    
        return [x, DELIMITER]

class BothDelimiter(Delimiter):
    def parts(self, x: T) -> List[Union[str, T]]:        
        return [DELIMITER, x, DELIMITER]

class NullDelimiter(Delimiter):
    def parts(self, x: T) -> List[T]:
        return [x]

def restorer(x: Sequence) -> Callable[[Iterable], Sequence]:
    if isinstance(x, str):
        return "".join
    else:
        return type(x)

def strip(xs: Sequence[T], y: T) -> Sequence[T]:
    """ Like str.strip but for any sequence. """
    result = xs
    if xs[0] == y:
        result = result[1:]
    if result[-1] == y:
        result = result[:-1]
    return result

def sequence_transformer(f):
    """ Return f' which applies f to a sequence preserving type. """
    def wrapped(s, *a, **k):
        restore = restorer(s)
        result = f(s, *a, **k)
        return restore(result)
    return wrapped

def padded_sliding(s: Iterable[T], window_size: int) -> Iterable[Tuple[T]]:
    assert window_size > 0
    n = len(s)
    for i in range(n + window_size - 1):
        left_padding = max(0, window_size - 1 - i)
        right_padding = max(0, (i + 1) - n)
        window = s[max(0, i - (window_size - 1)): min(n, i + 1)]
        yield left_padding, window, right_padding

def test_padded_sliding():
    assert list(padded_sliding("cat", 1)) == [(0, 'c', 0), (0, 'a', 0), (0, 't', 0)]
    assert list(padded_sliding("cat", 2)) == [(1, 'c', 0), (0, 'ca', 0), (0, 'at', 0), (0, 't', 1)]
    assert list(padded_sliding("cat", 3)) == [(2, 'c', 0), (1, 'ca', 0), (0, 'cat', 0), (0, 'at', 1), (0, 't', 2)]
    assert list(padded_sliding("", 1)) == []
    assert list(padded_sliding("", 2)) == [(1, '', 1)]
    assert list(padded_sliding(tuple('cat'), 2)) == [(1, ('c',), 0), (0, ('c', 'a'), 0), (0, ('a', 't'), 0), (0, ('t',), 1)]
    
def delimited_sequence_transformer(f):
    """ Return f' which applies f to a sequence preserving type and delimitation. """
    def wrapped(s, *a, **k):
        restore = restorer(s)
        l = list(s)
        has_left_delimiter = l[0] == DELIMITER
        has_right_delimiter = l[-1] == DELIMITER
        l2 = list(strip(s, DELIMITER))        
        r = f(l2, *a, **k)
        if has_left_delimiter:
            r = itertools.chain([DELIMITER], r)
        if has_right_delimiter:
            r = itertools.chain(r, [DELIMITER])
        return restore(r)
    return wrapped

def test_delimited_sequence_transformer():
    one = "abc"
    two = "def#"
    three = "#ghi"
    four = "#jkl#"
    
    f = delimited_sequence_transformer(lambda x: (x[1], x[2], x[0]))
    assert f(one) == "bca"
    assert f(two) == "efd#"
    assert f(three) == "#hig"
    assert f(four) == "#klj#"
    assert f(list(one)) == list("bca")
    assert f(list(two)) == list("efd#")
    assert f(list(three)) == list("#hig")
    assert f(list(four)) == list("#klj#")
    assert f(tuple(one)) == tuple("bca")
    assert f(tuple(two)) == tuple("efd#")
    assert f(tuple(three)) == tuple("#hig")
    assert f(tuple(four)) == tuple("#klj#")    

def test_delimiters():
    r = RightDelimiter()
    l = LeftDelimiter()
    b = BothDelimiter()
    n = NullDelimiter()

    assert r.delimit("abc") == "abc#"
    assert l.delimit("abc") == "#abc"
    assert b.delimit("abc") == "#abc#"
    assert n.delimit("abc") == "abc"

    assert r.delimit(tuple("abc")) == tuple("abc#")
    assert l.delimit(tuple("abc")) == tuple("#abc")
    assert b.delimit(tuple("abc")) == tuple("#abc#")
    assert n.delimit(tuple("abc")) == tuple("abc")

    assert r.delimit(list("abc")) == list("abc#")
    assert l.delimit(list("abc")) == list("#abc")
    assert b.delimit(list("abc")) == list("#abc#")
    assert n.delimit(list("abc")) == list("abc")

    assert r.delimit_string("abc") == "abc#"
    assert l.delimit_string("abc") == "#abc"
    assert b.delimit_string("abc") == "#abc#"
    assert n.delimit_string("abc") == "abc"

    assert r.delimit_sequence(tuple("abc")) == tuple("abc#")
    assert l.delimit_sequence(tuple("abc")) == tuple("#abc")
    assert b.delimit_sequence(tuple("abc")) == tuple("#abc#")
    assert n.delimit_sequence(tuple("abc")) == tuple("abc")

    assert r.delimit_sequence(list("abc")) == tuple("abc#")
    assert l.delimit_sequence(list("abc")) == tuple("#abc")
    assert b.delimit_sequence(list("abc")) == tuple("#abc#")
    assert n.delimit_sequence(list("abc")) == tuple("abc")

    import pandas as pd
    strings = pd.Series(["abc"]*10000)
    assert (r.delimit_array(strings) == pd.Series(["abc#"]*10000)).all()
    assert (l.delimit_array(strings) == pd.Series(["#abc"]*10000)).all()
    assert (b.delimit_array(strings) == pd.Series(["#abc#"]*10000)).all()
    assert (n.delimit_array(strings) == strings).all()

if __name__ == '__main__':
    import nose
    nose.runmodule()
