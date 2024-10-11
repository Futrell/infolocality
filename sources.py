import itertools
import random
from collections import Counter

import numpy as np
import scipy.special
import scipy.stats
import scipy.optimize
import pandas as pd
import tqdm

import utils

MK_PATH = "data/generated-MI-distros/generated-MI-distros_%s.txt"

def log_rem(*shape, T=1):
    return scipy.special.log_softmax(1/T*np.random.randn(*shape))

def rem(*shape, T=1, lam=1):
    return scipy.special.softmax(lam*1/T*np.random.randn(*shape))

def zipf_mandelbrot(N, s=1, q=0):
    k = np.arange(float(N)) + 1
    p = (k + q) ** -s
    Z = p.sum()
    return p/Z

def log_zipf_mandelbrot(N, s=1, q=0):
    k = np.arange(N) + 1
    energy = s * np.log(k+q)
    lnZ = scipy.special.logsumexp(-energy)
    return -energy - lnZ

def factor(p, m, k):
    return p.reshape(*(m,)*k)

def product_distro(p1, p2):
    return np.outer(p1, p2).flatten()

def log_product_distro(lp1, lp2):
    return lp1[:, None] + lp2[None, :]

def flip(p):
    return np.array([p, 1-p])

def log_flip(p):
    return np.log(np.array([p, 1-p]))

def mi_mix(pi):
    """ Generate a 2x2 distribution with MI monotonically related to pi. """
    A = np.ones(4)/2
    A[0b00] *= pi*1 + (1-pi)*1/2  # 00
    A[0b01] *= pi*0 + (1-pi)*1/2  # 01
    A[0b10] *= pi*0 + (1-pi)*1/2  # 10
    A[0b11] *= pi*1 + (1-pi)*1/2  # 11
    return A

def mi_mix3(i12, i23):
    A = np.ones(8)/2 # start with 1/2 everywhere
    A[0b000] *= (i12*1 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2) 
    A[0b001] *= (i12*1 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b010] *= (i12*0 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b011] *= (i12*0 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2)
    A[0b100] *= (i12*0 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2) 
    A[0b101] *= (i12*0 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b110] *= (i12*1 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b111] *= (i12*1 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2)
    return A

entropy = scipy.stats.entropy

def mi(p_xy):
    """ Calculate mutual information of a distribution P(x,y) 

    Input: 
    p_xy: An X x Y array giving p(x,y)
    
    Output:
    The mutual information I[X:Y], a nonnegative scalar,
    """
    p_x = p_xy.sum(axis=-1)
    p_y = p_xy.sum(axis=-2)
    return entropy(p_x, axis=None) + entropy(p_y, axis=None) - entropy(p_xy, axis=None)

def tc(joint):
    marginals = 0.0
    for i in range(joint.ndim):
        axes = tuple([j for j in range(joint.ndim) if j != i])
        marginal = joint.sum(axes)
        marginals += entropy(marginal)
    return marginals - entropy(joint, axis=None)

def subsets(xs):
    return itertools.chain.from_iterable(
        itertools.combinations(xs, r)
        for r in range(len(xs)+1)
    )

def coinformation_lattice(joint):
    # joint is a KxKxK... array of probabilities.
    T = joint.ndim
    entropies = {}
    for indices in subsets(range(T)):
        complement = [i for i in range(T) if i not in indices]
        marginal = joint.sum(axis=tuple(complement))
        entropies[indices] = entropy(marginal, axis=None)
    q = -(-1)**np.arange(T+1)
    coinformations = {
        indices : sum(q[len(subset)] * entropies[subset] for subset in subsets(indices))
        for indices in entropies
    }
    return coinformations

def test_coinformation_lattice():
    # Synergy pattern
    xor = np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]]
    ])/4
    lattice = coinformation_lattice(xor)
    results = [
        lattice[()], lattice[(1,)], lattice[(2,)], lattice[(2,)], lattice[0,1], lattice[0,2], lattice[1,2], lattice[0,1,2]
    ]
    assert np.allclose(results, [0, np.log(2), np.log(2), np.log(2), 0, 0, 0, -np.log(2)])

    # Redundancy pattern
    bigbit = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])/2
    lattice = coinformation_lattice(bigbit)
    results = [
        lattice[()], lattice[(1,)], lattice[(2,)], lattice[(2,)], lattice[0,1], lattice[0,2], lattice[1,2], lattice[0,1,2]
    ]
    assert results[0] == 0
    assert np.allclose(results[1:], np.log(2))

    # Markov chain can never have negative coinformation
    for i in range(2,10):
        markov = scipy.special.softmax(
            np.random.randn(i,i,1,1,1,1) +
            np.random.randn(1,i,i,1,1,1) +
            np.random.randn(1,1,i,i,1,1) +
            np.random.randn(1,1,1,i,i,1) +
            np.random.randn(1,1,1,1,i,i)
        )
        assert all(mi >= -0.0000001 for mi in coinformation_lattice(markov).values())

    # 3-Parity
    parity3 = np.array([
        [[[1, 0],    # 000X
          [0, 1]],   # 001X
         [[0, 1],    # 010X
          [1, 0]]],  # 011X
        [[[0, 1],    # 100X
          [1, 0]],   # 101X
         [[1, 0],    # 110X
          [0, 1]]]   # 111X
    ])/8
    lattice = coinformation_lattice(parity3)
    univariate_results = [lattice[(0,)], lattice[(1,)], lattice[(2,)], lattice[(3,)]]
    assert np.allclose(univariate_results, np.log(2))
    assert np.allclose(lattice[(0,1,2,3)], np.log(2))
    for indices in lattice:
        if len(indices) == 0 or len(indices) == 2 or len(indices) == 3:
            assert np.allclose(lattice[indices], 0)

def perturbed_conditional(px, py, perturbation_size=1):
    """
    p(y | x) \propto p(y) exp(perturbation_size * a(x,y)) where a(x,y) ~ Normal.
    Shape X x Y.
    """
    a = np.random.randn(len(px), len(py))
    energy = np.log(py)[None, :] + perturbation_size*a
    return scipy.special.softmax(energy, -1)

def mostly_independent(*Vs, coupling=1, source='zipf', consistent=True, systematic=True, **kwds):
    # first make marginal distributions for each V
    if source == 'zipf': 
        ps = [zipf_mandelbrot(V, **kwds) for V in Vs]
    elif source == 'rem':
        if consistent:
            V = utils.the_unique(Vs)
            p = rem(V, **kwds)
            ps = [p for _ in range(len(Vs))]
        else:
            ps = [rem(V, **kwds) for V in Vs]
    # now make a joint distribution over V*
    source = 1
    for p in ps:
        source = np.outer(source, p)
    source = scipy.special.softmax(np.log(source.reshape(Vs)) + coupling*np.random.randn(*Vs))
    meanings = itertools.product(*[range(V) for V in Vs])
    return source, list(meanings)
    
def dependents(V, k, coupling=1, source='rem', consistent=True, **kwds):
    """
    A distribution of head with k dependents.
    
    V - number of distinct words.
    k - number of dependents
    
    """
    # meaning of V and k is messed up -- should be V^k, but is V*k.
    if source == 'zipf':
        h = zipf_mandebelbrot(V, **kwds)
        ds = [zipf_mandelbrot(V, **kwds) for _ in range(k)]
    elif source == 'rem':
        h = rem(V, **kwds)
        if consistent:
            d = rem(V, **kwds)
            ds = np.array([d for _ in range(k)])
        else:
            ds = np.array([rem(V, **kwds) for _ in range(k)])       
    conditionals = np.array([
        perturbed_conditional(h, d, perturbation_size=coupling*np.sqrt(V)/(i+1))
        for i, d in enumerate(ds)
    ])
    mis = [mi(h[:, None] * conditional) for conditional in conditionals]
    def gen():
        for configuration in utils.cartesian_indices(V, k+1):
            d = {'h': ('h', configuration[0])}
            d |= {'d%d' % i : d for i, d in enumerate(configuration[1:])}
            p = h[configuration[0]]
            for i, dep in enumerate(configuration[1:]):
                p *= conditionals[i, configuration[0], dep]
            yield d, p
    meanings, ps = zip(*gen())
    return ps, meanings, mis

def astarn(num_A, num_N, num_classes=1, p_halt=.5, maxlen=4, source='rem', coupling=1, **kwds):
    import einops
    # UGH! Needs to be a noun plus a MULTISET of adjectives, unordered...
    if source == 'zipf':
        As = [zipf_mandelbrot(num_A, **kwds) for _ in range(num_classes)]
        N = zipf_mandelbrot(num_N, **kwds)
    elif source == 'rem':
        As = rem(num_classes, num_A, **kwds)
        N = rem(num_N, **kwds)
    ANs = np.array([
        perturbed_conditional(N, A, perturbation_size=coupling*np.sqrt(num_A)/(k+1))
        for k, A in enumerate(As)
    ]) # shape C x N x A
    AN = N[None, :] * einops.rearrange(ANs, "c n a -> (c a) n") / num_classes # shape AN
    A = AN.sum(axis=-1, keepdims=True)
    pmi = einops.rearrange(np.log(AN) - np.log(A) - np.log(N), "(c a) n -> c n a", c=num_classes, a=num_A)
    class_mis = [mi(N[:, None] * AN) for AN in ANs]   
    def gen():
        for n in range(num_N):
            d = {'n': n, 'p': N[n]}
            for k in range(maxlen):
                p_k = (1-p_halt)**k * (p_halt if k <= maxlen else 1) / num_classes
                for classes in utils.cartesian_indices(num_classes, k):
                    for adjectives in utils.cartesian_indices(num_A, k):
                        d = {'n': (n,)}
                        d |= {str(i) : (c,a) for i, (c, a) in enumerate(zip(classes, adjectives))}
                        p = N[n] * p_k * np.prod([ANs[c,n,a] for c, a in zip(classes, adjectives)])
                        pmi_d = {str(i) : pmi[c,a,n] for i, (c,a) in enumerate(zip(classes, adjectives))}
                        yield p, d, pmi_d
    ps, meanings, pmis = zip(*gen())
    # want conditional entropy of nouns given each adjective
    ps = np.array(ps)
    ps = ps / np.sum(ps)
    return ps, meanings, pmis, class_mis

def mansfield_kemp_source(which='training', A_rate=1, N_rate=1, D_rate=1): # or 'test'
    # Noun Adj Num Dem, disjoint, represented as tuples
    filename = MK_PATH % which
    df = pd.read_csv(filename, sep="\t")
    df.columns = "n A N D".split()
    counts = Counter(tuple(dict(row).items()) for _, row in df.iterrows())
    def gen():
        rates = [A_rate, N_rate, D_rate]
        for meaning, count in counts.items():
            modifiers = [meaning[1], meaning[2], meaning[3]]
            for mask in itertools.product(*[[0,1]]*3):
                truncated = (meaning[0],) + tuple([m for i, m in enumerate(modifiers) if mask[i]])
                trunc_prob = count
                for i, rate in enumerate(rates):
                    trunc_prob *= rate if mask[i] else (1-rate)
                if trunc_prob > 0:
                    yield truncated, trunc_prob
    meanings, ps = zip(*gen())
    ps = np.array(ps)
    ps = ps / ps.sum()
    return ps, meanings

def empirical_source(filename, truncate=0, len_limit=0, rename=None, filters=None):
    df = pd.read_csv(filename)
    def gen():
        for i, row in df.iterrows():
            meaning = []
            for key, value in row.items():
                if not pd.isna(value):
                    if rename:
                        meaning.append((rename[key], value))
                    else:
                        meaning.append((key, value))
            if len(meaning) > len_limit:
                yield tuple(meaning)
    counts = Counter(gen())
    if truncate:
        counts = {k:v for k, v in counts.items() if v > truncate}
    meanings, ps = zip(*counts.items())
    ps = np.array(ps)
    ps = ps / ps.sum()
    return ps, meanings

def couple(p, q, coupling=.5, coupling_type='logspace'):
    """ Couple distribution p with q, where resulting distribution has same entropy as p """
    if coupling_type == 'mixture':
        mix = np.log((1-coupling)*p + coupling*q)
    else:
        mix = (1-coupling) * np.log(p) + coupling * np.log(q)
    target = entropy(p)
    def objective(T):
        value = entropy(scipy.special.softmax(T*mix))
        return ((value - target)**2).sum()
    T = scipy.optimize.minimize(objective, 1).x.item()
    return scipy.special.softmax(T*mix)

def hierarchical_source(two=.99, three=.2, six=.01, V=5, shuffle=True, **kwds):
    atom = zipf_mandelbrot(V, **kwds)
    V2_to_V = np.eye(V)[list(range(V))*V]
    V2_to_V /= V2_to_V.sum()

    permute = np.random.permutation if shuffle else lambda x:x
    
    p1o = (1 - two) * product_distro(atom, atom).reshape(V,V) + two * permute(np.eye(V))/V
    p1 = (1 - three) * product_distro(p1o.flatten(), atom).reshape(V**2, V) + three * permute(V2_to_V)

    p2o = (1 - two) * product_distro(atom, atom).reshape(V,V) + two * permute(np.eye(V))/V
    p2 = (1 - three) * product_distro(p2o.flatten(), atom).reshape(V**2, V) + three * permute(V2_to_V)

    p = (1 - six) * product_distro(p1.flatten(), p2.flatten()) + six * permute(np.eye(V**3).flatten())/V**3

    return p.reshape(V,V,V,V,V,V)

if __name__ == '__main__':
    import nose
    nose.runmodule()
