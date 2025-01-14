""" Probability monad """
import random
from collections import Counter, namedtuple, deque
from math import log, exp
import operator
import functools
import itertools

import numpy as np
import scipy.special
import scipy.stats
import pandas as pd
import sympy
import rfutils

INF = float('inf')
_SENTINEL = object()

def keep_calling_forever(f):
    return iter(f, _SENTINEL)

# safelog : Float -> Float
def safelog(x):
    if x == 0:
        return -INF
    else:
        return log(x)

def identity(x):
    return x

# logaddexp : Float x Float -> Float
def logaddexp(one, two):
    return safelog(exp(one) + exp(two))

def logsubexp(one, two):
    return log(exp(one) - exp(two))

def np_logaddexp(one, two):
    return np.log(np.exp(one) + np.exp(two))

def np_logsubexp(one, two):
    return np.log(np.exp(one) - np.exp(two))

# logsumexp : [Float] -> Float
def logsumexp(xs):
    return safelog(sum(map(exp, xs)))

# reduce_by_key : (a x a -> a) x [(b, a)] -> {b -> a}
def reduce_by_key(f, keys_and_values):
    d = {}
    for k, v in keys_and_values:
        if k in d:
            d[k] = f(d[k], v)
        else:
            d[k] = v
    return d

def lazy_product_map(f, xs):
    """ equivalent to itertools.product(*map(f, xs)), but does not hold the values
    resulting from map(f, xs) in memory. xs must be a sequence. """
    if not xs:
        yield []
    else:
        x = xs[0]
        for result in f(x):
            for rest in lazy_product_map(f, xs[1:]):
                yield [result] + rest

class Monad(object):
    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.values) + ")"

    def __rshift__(self, f):
        return self.bind(f)

    def __add__(self, bindee_without_arg):
        return self.bind(lambda _: bindee_without_arg())

    # lift_ret : (a -> b) -> (a -> m b)
    @classmethod
    def lift_ret(cls, f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            return cls.ret(f(*a, **k))
        return wrapper

    @property
    def mzero(self):
        return type(self)(self.zero)

    @classmethod
    def guard(cls, truth):
        """
        Usage:
        lambda x: Monad.guard(condition(x)) >> (lambda _: consequent(x))
        """
        if truth:
            return cls.ret(_SENTINEL) # irrelevant value
        else:
            return cls(cls.zero) # construct mzero

    @classmethod
    def mapM(cls, mf, xs):
        # Reference implementation to be overridden by something more efficient.
        def f(acc, x):
            # f acc x = do
            # r <- mf(x);
            # return acc + (r,)
            return mf(x).bind(lambda r: cls.ret(acc + (r,)))
        return cls.reduceM(f, xs, initial=())

class Amb(Monad):
    def __init__(self, values):
        self.values = values

    zero = []

    def sample(self):
        return next(iter(self))

    def bind(self, f):
        return Amb(rfutils.flatmap(f, self.values))

    @classmethod
    def ret(cls, x):
        return cls([x])

    def __iter__(self):
        return iter(self.values)

    # mapM : (a -> Amb b) x [a] -> Amb [b]
    @classmethod
    def mapM(cls, f, *xss):
        return Amb(itertools.product(*map(f, *xss)))

    # filterM : (a -> Amb Bool) x [a] -> Amb [a]
    @classmethod
    def filterM(cls, f, xs):
        return cls(itertools.compress(xs, mask) for mask in cls.mapM(f, xs))

    # reduceM : (a x a -> Amb a) x [a] -> Amb [a]
    @classmethod
    def reduceM(cls, f, xs, initial=None):
        def do_it(acc, xs):
            if not xs:
                yield acc
            else:
                x = xs[0]
                xs = xs[1:]
                for new_acc in nf(acc, x):
                    for res in do_it(new_acc, xs):
                        yield res
        xs = tuple(xs)
        if initial is None:
            return cls(do_it(xs[0], xs[1:]))
        else:
            return cls(do_it(initial, xs))

    def conditional(self, f=None, normalized=True):
        if f is None:
            f = lambda x: x

        class CDict(dict):
            def __missing__(d, key):
                samples = (y for x, y in map(f, self.values) if x == key)
                d[key] = Amb(samples)
                return d[key]

        return CDict()

def Samples(rf):
    return Amb(keep_calling_forever(rf))

Field = namedtuple('Field',
     ['add', 'sub', 'sum', 'mul', 'div', 'zero', 'one', 'to_log', 'to_p', 'from_log', 'from_p', 'pow']
)
p_space = Field(
    operator.add, operator.sub, sum, operator.mul, operator.truediv, 0, 1, safelog, identity, exp, identity, operator.pow,
)
log_space = Field(
    logaddexp, logsubexp, logsumexp, operator.add, operator.sub, -INF, 0, identity, exp, identity, safelog, operator.mul,
)
numpy_p_space = Field(
    operator.add, operator.sub, np.sum, operator.mul, operator.truediv, 0, 1, np.log, identity, np.exp, identity, operator.pow,
)
numpy_log_space = Field(
    np_logaddexp, np_logsubexp, scipy.special.logsumexp, operator.add, operator.sub, -INF, 0, identity, np.exp, identity, np.log, operator.mul,
)

class Sample(Monad):
    def __init__(self, values):
        self.values = values
    
    def bind(self, f):
        return type(self)(f(self.values))


class Enumeration(Monad):
    def __init__(self,
                 values,
                 marginalized=False,
                 normalized=False):
        self.marginalized = marginalized
        self.normalized = normalized
        self.values = values
        if isinstance(values, dict):
            self.values = values.items()
            self._dict = values
        else:
            self.values = values
            self._dict = None

    field = log_space
    zero = []

    def sample(self):
        xs, ps = zip(*self.values)
        return np.random.choice(range(len(xs)), p=list(map(self.field.to_p, ps)))

    # Ma x (a -> Mb) -> Mb
    def bind(self, f):
        mul = self.field.mul
        vals = [
            (y, mul(p_y, p_x))
            for x, p_x in self.values
            for y, p_y in f(x)
        ]
        return type(self)(vals).marginalize().normalize()

    # return : a -> Enum a
    @classmethod
    def ret(cls, x):
        return cls(
            [(x, cls.field.one)],
            normalized=True,
            marginalized=True,
        )

    def marginalize(self):
        if self.marginalized:
            return self
        else:
            # add together probabilities of equal values
            result = reduce_by_key(self.field.add, self.values)
            # remove zero probability values
            zero = self.field.zero
            result = {k:v for k, v in result.items() if v != zero}
            return type(self)(
                result,
                marginalized=True,
                normalized=self.normalized,
            )

    def normalize(self):
        """ """
        if self.normalized:
            return self
        else:
            enumeration = list(self.values)
            Z = self.field.sum(p for _, p in enumeration)
            div = self.field.div
            result = [(thing, div(p, Z)) for thing, p in enumeration]
            return type(self)(
                result,
                marginalized=self.marginalized,
                normalized=True,
            )

    def __iter__(self):
        return iter(self.values)

    @property
    def dict(self):
        if self._dict:
            return self._dict
        else:
            self._dict = dict(self.values)
            return self._dict

    def __getitem__(self, key):
        return self.dict[key]

    @classmethod
    def mapM(cls, ef, *xss):
        mul = cls.field.mul
        one = cls.field.one
        def gen():
            for sequence in itertools.product(*map(ef, *xss)):
                if sequence:
                    seq, ps = zip(*sequence)
                    yield tuple(seq), functools.reduce(mul, ps, one)
                else:
                    yield tuple(), one
        return cls(gen()).marginalize().normalize()

    @classmethod
    def reduceM(cls, ef, xs, initial=None):
        mul = cls.field.mul
        one = cls.field.one
        def do_it(acc, xs):
            if not xs:
                yield (acc, one)
            else:
                the_car = xs[0]
                the_cdr = xs[1:]
                new_acc_distro = ef(acc, the_car).marginalize().normalize()
                for new_acc, p in new_acc_distro:
                    for res, p_res in do_it(new_acc, the_cdr):
                        yield res, mul(p, p_res)
        xs = tuple(xs)
        if initial is None:
            result = do_it(xs[0], xs[1:])
        else:
            result = do_it(initial, xs)
        return cls(result).marginalize().normalize()

    def expectation(self, f):
        """ E[f(x)] """
        return sum(exp(lp)*f(v) for v, lp in self.values)

    def exponentiate(self, a):
        """ q(x) = 1/Z p(x)^a """
        pow = self.field.pow
        return type(self)((value, pow(lp, a)) for value, lp in self.values).normalize()

    def entropy(self):
        """ Entropy of the distribution """
        ps = [p for _, p in self.normalized()]
        return scipy.stats.entropy(self.field.to_p(ps)) / log(2)

    def mutual_information(self):
        """ Only works if support is over sequences of length 2. """
        m1 = self >> self.lift_ret(operator.itemgetter(0))
        m2 = self >> self.lift_ret(operator.itemgetter(1))
        return m1.entropy() + m2.entropy() - self.entropy()

    def conditional(self, f=None, normalized=True):
        if f is None:
            f = lambda x: x

        add = self.field.add
        d = {}
        for value, p in self.values:
            condition, outcome = f(value)
            if condition in d:
                if outcome in d[condition]:
                    d[condition][outcome] = add(d[condition][outcome], p)
                else:
                    d[condition][outcome] = p
            else:
                d[condition] = {outcome: p}
        cls = type(self)
        if normalized:
            return {
                k : cls(v).normalize()
                for k, v in d.items()
            }
        else:
            return {k: cls(v) for k, v in d.items()}

    @classmethod
    def flip(cls, p):
        """ A Bernoulli distribution with probability p. """
        pp = cls.field.from_p(p)
        return cls(
            [(True, pp), (False, cls.field.sub(cls.field.one, pp))],
            marginalized=True,
            normalized=True,
        )

    @classmethod
    def uniform(cls, xs):
        """ A uniform distribution over the xs """
        #return cls((x, cls.field.one) for x in xs).normalize()
        # faster:
        xs = list(xs)
        p = cls.field.div(cls.field.one, cls.field.from_p(len(xs)))
        return cls([(x, p) for x in xs], normalized=True)

    @classmethod
    def exponential(cls, xs, alpha=1):
        """ p(x_k) \propto e^{-\alpha k} """
        return cls((x, cls.field.from_p(np.exp(-alpha*t))) for t, x in enumerate(xs)).normalize()

    @classmethod
    def zipf_mandelbrot(cls, xs, alpha=1, q=0):
        """ p(x_k) \propto (k+q)^{-\alpha} """
        return cls((x, cls.field.from_p((t+q)**-alpha)) for t, x in enumerate(xs)).normalize()

    def incrementalize(self, t, delimiter='#'):
        """ Transform a distribution over strings p(x) into a joint distributions
        over contexts of length t and next characters. Note a correction factor:
        longer sequences have more beginning probability.
        """
        mul = self.field.mul
        from_p = self.field.from_p
        unnormalized = type(self)( 
            (x, mul(from_p(len(x)+1), p))
            for x, p in self.values
        )
        return unnormalized >> (lambda x: self.uniform(increments(x + delimiter, t)))

class PSpaceEnumeration(Enumeration):
    field = p_space

    def entropy(self):
        ps = [p for _, p in self.normalize()]
        return np.sum(scipy.special.entr(ps)) / log(2)

    def expectation(self, f):
        return sum(p*f(v) for v, p in self.values)

class NpEnumeration(Enumeration):
    field = numpy_log_space
    
    def __init__(self, xs, ps, marginalized=False, normalized=False):
        self.marginalized = marginalized
        self.normalized = normalized
        self.xs = pd.Series(xs)
        self.ps = pd.Series(ps)

    def bind(self, f): # (a -> m b) x a -> m b
        if len(self.xs) == 0:
            return self.mzero
        else:
            y_monads = self.xs.map(f)
            ys = pd.concat(y_monads.map(operator.attrgetter('xs')).to_list())
            ps = pd.concat(y_monads.map(operator.attrgetter('ps')).to_list())
            return type(self)(ys, ps).marginalize().normalize()

    @classmethod
    def ret(cls, x):
        return cls(pd.Series([x]), pd.Series([cls.field.one]), normalized=True, marginalized=True)

    def marginalize(self):
        if self.marginalized:
            return self
        else:
            df = self.ps.groupby(self.xs).agg(self.field.sum).reset_index()
            return type(self)(df['index'], df[0], normalized=self.normalized, marginalized=True)

    def normalize(self):
        if self.normalized:
            return self
        else:
            Z = self.field.sum(self.ps)
            q = self.field.div(self.ps, Z)
            return type(self)(self.xs, q, marginalized=self.marginalized, normalized=True)

    @property
    def values(self):
        return list(zip(self.xs, self.ps))

    @property
    def dict(self):
        return dict(self.values)

    def expectation(self, f):
        return self.field.sum(self.field.mul(self.xs.map(f), self.ps))

    @classmethod
    def mapM(cls, ef, xs): # (a -> m b) x [a] -> m [b]
        y_monads = [ef(x) for x in xs]
        y = itertools.product(*[y_monad.xs for y_monad in y_monads])
        p = np.array([cls.field.one])
        for y_monad in y_monads:
            p = cls.field.mul(p[:, None], y_monad.ps[None, :]).flatten()
        data = pd.DataFrame({'x': y, 'p': p})
        return cls(data)

    @classmethod
    def reduceM(cls, ef, xs, initial=None):
        ...

    def exponentiate(self, a):
        q = self.field.pow(self.ps, a)
        data = pd.DataFrame({'x': self.xs, 'p': q})
        return type(self)(data, marginalized=self.marginalized).normalize()

    def entropy(self, f=None):
        ps = self.field.to_p(self.ps)
        return scipy.stats.entropy(ps) / log(2)

    @classmethod
    def flip(cls, p):
        pp = cls.field.from_p(p)
        data = pd.DataFrame({'x': [True, False], 'p': [pp, cls.field.sub(cls.field.one, pp)]})
        return cls(data)

    @classmethod
    def exponential(cls, xs, alpha=1):
        """ p(x_k) \propto e^{-\alpha k} """
        return cls(xs, [cls.field.from_p(np.exp(-alpha*t)) for t, x in enumerate(xs)]).normalize()

    @classmethod
    def uniform(cls, xs):
        xs = np.array(xs)
        ps = cls.field.div(cls.field.one, cls.field.from_p(len(xs)))
        return cls(pd.Series(xs), ps, normalized=True)

class SymbolicEnumeration(PSpaceEnumeration):

    def marginalize(self):
        result = super().marginalize()
        new_result = {k:sympy.simplify(v) for k, v in result.values}
        return type(result)(
            new_result,
            marginalized=True,
            normalized=result.normalized
        )

    def entropy(self):
        return -sum(p*sympy.log(p) for _, p in self.normalize())

def increments(xs, t, padding='#'):
    # t is context length
    context = deque(padding*t, maxlen=t)
    for x in xs:
        yield "".join(context), x
        context.append(x)

def UniformEnumeration(xs):
    xs = list(xs)
    N = len(xs)
    return Enumeration([(x, -log(N)) for x in xs])

def UniformSamples(xs):
    return Samples(lambda: random.choice(xs))

def enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return Enumeration(f(*a, **k))
    return wrapper

def pspace_enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return PSpaceEnumeration(f(*a, **k))
    return wrapper

def uniform_enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return UniformEnumeration(f(*a, **k))
    return wrapper

deterministic = Enumeration.lift_ret
certainly = Enumeration.ret

def bayes(prior, likelihood, observed):
    """ Probability of h given p(h) and f(h) == observed """
    joint = prior >> (lambda h: likelihood(h) >> (lambda x: prior.ret((h,x))))
    return joint >> (lambda pair:
        prior.guard(pair[-1] == observed) >> (lambda _:
        prior.ret(pair[0])))

def uniform(xs):
    xs = list(xs)
    n = len(xs)
    return Enumeration([(x, -log(n)) for x in xs])

def sampler(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return Samples(lambda: f(*a, **k))
    return wrapper

def enumeration_from_samples(samples, num_samples):
    counts = Counter(itertools.islice(samples, None, num_samples))
    return Enumeration((k, log(v)) for k, v in counts.items()).normalize()

def enumeration_from_sampling_function(f, num_samples):
    samples = iter(f, _SENTINEL)
    return enumeration_from_samples(samples, num_samples)

def approx_enumerator(num_samples):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            sample_f = lambda: f(*a, **k)
            return enumeration_from_sampling_function(sample_f, num_samples)
        return wrapper
    return decorator

# enum_flip :: Float -> Enum Bool
@enumerator
def enum_flip(p):
    if p > 0:
        yield True, log(p)
    if p < 1:
        yield False, log(1-p)

@pspace_enumerator
def pspace_flip(p):
    if p > 0:
        yield True, p
    elif p < 1:
        yield False, 1 - p

def surprisal(distribution, value):
    return -distribution.field.to_log(distribution.dict[value])

def test_pythagorean_triples():
    n = 25
    result = uniform(range(1, n+1)) >> (lambda x: # x ~ uniform(1:n);
             uniform(range(x+1, n+1)) >> (lambda y: # y ~ uniform(x+1:n);
             uniform(range(y+1, n+1)) >> (lambda z: # z ~ uniform(y+1:n);
             Enumeration.guard(x**2 + y**2 == z**2) >> (lambda _: # constraint
             Enumeration.ret((x,y,z)))))) # return a triple deterministically
    assert set(result.dict.keys()) == {
        (3, 4, 5),
        (5, 12, 13),
        (6, 8, 10),
        (7, 24, 25),
        (8, 15, 17),
        (9, 12, 15),
        (12, 16, 20),
        (15, 20, 25)
    }
    assert all(logp == -2.079441541679836 for logp in result.dict.values())

def send_more_money():
    # Nice as an example but way too slow to be a test.
    
    def encode(*xs):
        return sum(10**i * x for i, x in enumerate(reversed(xs)))
    
    result = uniform(range(10)) >> (lambda s:
             uniform(range(10)) >> (lambda e:
             uniform(range(10)) >> (lambda n:
             uniform(range(10)) >> (lambda d:
             uniform(range(10)) >> (lambda m:
             uniform(range(10)) >> (lambda o:
             uniform(range(10)) >> (lambda r:
             uniform(range(10)) >> (lambda y:
             Enumeration.guard(
                 encode(s,e,n,d) + encode(m,o,r,e) == encode(m,o,n,e,y)
             ) >> (lambda _: Enumeration.ret((s,e,n,d,m,o,r,y)))))))))))

    assert result == (9, 5, 6, 7, 1, 0, 8, 2)
        
if __name__ == '__main__':
    import nose
    nose.runmodule()
