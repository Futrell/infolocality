""" Probabilistic context-free rewriting systems in the probability monad """
from collections import namedtuple, Counter
from math import log, exp
import functools
import operator

import rfutils
import pyrsistent as pyr

from pmonad import *

Rule = namedtuple('Rule', ['lhs', 'rhs'])
def concatenate(sequences):
    return sum(sequences, ())

def make_pcfg(monad, rules, start='S'):
    rewrites = process_pcfg_rules(monad, rules)
    return PCFG(monad, rewrites, start)

def make_bounded_pcfg(monad, rules, bound, start='S'):
    rewrites = process_pcfg_rules(monad, rules)
    return BoundedPCFG(monad, rewrites, start, bound)

def process_pcfg_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        prob = monad.field.from_p(prob)
        if rule.lhs in d:
            d[rule.lhs].append((rule.rhs, prob))
        else:
            d[rule.lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

def process_gpsg_rules(monad, rules):
    return process_pcfg_rules(monad, expand_gspg_rules(rules))

def dict_product(d):
    """ {k:[v]} -> [{k:v}] """
    canonical_order = sorted(d.keys())
    value_sets = [d[k] for k in canonical_order]
    assignments = itertools.product(*value_sets)
    for assignment in assignments:
        yield dict(zip(canonical_order, assignment))

def dict_subset(d, ks):
    return {k:d[k] for k in ks}

# [(a,b)] -> {a:{b}}
def dict_of_sets(pairs):
    return rfutils.mreduce_by_key(set.add, pairs, set)

flat = itertools.chain.from_iterable

def expand_gpsg_rules(rules, **values):
    """ Compile GPSG-style rules into CFG rules.
    Syntax for a GPSG-style rule:
    A_{f} -> B_{f} C_{g} D_{f}
    {f} is a copied feature. This will be expanded into equiprobable rules of the form
    A_f:v -> B_f:v C_g:w D_f:v
    for all values v of f and w of g.
    """
    def free_variables_in(element):
        """ Iterate over free variables in a symbol. """
        parts = element.split("_")
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                yield part.strip("{}")

    def possible_feature_values_in(element):
        """ Iterate over tuples of feature labels and feature values in a symbol. """
        parts = element.split("_")
        for part in parts:
            if ":" in part:
                k, v = part.split(":")
                yield k, part 
                
    def possible_feature_values(rules):
        """ Get dictionary from feature labels to the set of feature values. """
        elements = flat((rule.lhs,) + rule.rhs for rule in rules)
        pairs = flat(map(possible_feature_values_in, elements))
        extracted = dict_of_sets(pairs)
        for label, vals in values.items():
            if label in extracted:
                extracted[label] |= {"%s:%s" % (label, val) for val in vals}
            else:
                extracted[label] = {"%s:%s" % (label, val) for val in vals}
        return extracted

    rules = list(rules) # we'll have to go through twice
    possibilities = possible_feature_values(rule for rule, _ in rules)
    for rule, prob in rules:
        free_variables = set(free_variables_in(rule.lhs))
        for element in rule.rhs:
            free_variables.update(free_variables_in(element))
        assignments = list(dict_product(
            dict_subset(possibilities, free_variables)
        ))
        Z = len(assignments)
        for assignment in assignments:
            new_lhs = rule.lhs.format_map(assignment)
            new_rhs = tuple(
                element.format_map(assignment) for element in rule.rhs
            )
            yield Rule(new_lhs, new_rhs), prob/Z

def test_expand_gpsg_rules():
    stuff = set(expand_gpsg_rules([
        (Rule('S', ('NP_g:f', 'VP_g:f')), .5),
        (Rule('S', ('NP_g:m', 'VP_g:m')), .5),
        (Rule('NP_{g}', ('A_{g}', 'N_{g}')), 1),
    ]))
    assert stuff == {
        (Rule(lhs='S', rhs=('NP_g:f', 'VP_g:f')), 0.5),
        (Rule(lhs='S', rhs=('NP_g:m', 'VP_g:m')), 0.5),
        (Rule(lhs='NP_g:f', rhs=('A_g:f', 'N_g:f')), 0.5),
        (Rule(lhs='NP_g:m', rhs=('A_g:m', 'N_g:m')), 0.5),
    }

    stuff2 = set(expand_gpsg_rules([
        (Rule('S', ('NP_{g}', 'VP_{g}')), 1),
        (Rule('NP_{g}', ('A_{g}', 'N_{g}')), 1),
    ], g=['m', 'f']))
    assert stuff2 == stuff

def german_nesting_rules(num_hsem=1, num_tsem=5, num_isem=1, num_asem=1):
    nonterminal_rules = [ #
        # Sentence rule                             
        Rule(lhs='S', rhs=('NP_case:nom_{asem}', 'VP_syn:mat_form:fin')),

        # Haben
        Rule(lhs='VP_syn:mat_form:fin', rhs=('hat', 'VP_syn:sub_form:part')), 
        Rule(lhs='VP_syn:sub_form:fin', rhs=('VP_syn:sub_form:part', 'hat')), # (dead rule?)

        # NP rule
        Rule(lhs='NP_case:acc_{tsem}', rhs=('Det_case:acc', 'N_{tsem}',)),
        Rule(lhs='NP_{case}_{asem}', rhs=('Det_{case}', 'N_{asem}',)),

        # Intransitive VP
        Rule(lhs='VP_{syn}_{form}', rhs=('Vi_{form}_{isem}',)),

        # Transitive VP
        Rule(lhs='VP_syn:mat_{form}', rhs=('Vt_{form}_{tsem}', 'NP_case:acc_{tsem}')), # Vt
        Rule(lhs='VP_syn:sub_{form}', rhs=('NP_case:acc_{tsem}', 'Vt_{form}_{tsem}')), # Vt

        # Causative VP
        Rule(lhs='VP_{syn}_{form}', rhs=('VPH_{syn}_{form}',)), # VPH is a subset of VP for any syn and form
        Rule(lhs='VPH_syn:sub_form:part', rhs=('VPH_syn:sub_form:inf',)), # final VPH can convert to infinitive if not finite
        # Nested causative:
        Rule(lhs='VPH_syn:mat_{form}', rhs=('Vh_{form}_{hsem}', 'NP_case:dat_{asem}', 'VP_syn:sub_form:inf')), 
        Rule(lhs='VPH_syn:sub_{form}', rhs=('NP_case:dat_{asem}', 'VP_syn:sub_form:inf', 'Vh_{form}_{hsem}')),

        # Non-nested causative? Complicated:
        
        # er hat der Frau die Kühe melken geholfen
        # er hat der Frau geholfen, die Kühe zu melken
        # er hat dem Lehrer [der Frau [die Kühe melken] lehren] geholfen
        # er hat dem Lehrer [der Frau lehren, die Kühe zu melken] geholfen
        # er hat dem Lehrer geholfen, der Frau die Kühe melken zu lehren
        # er hat dem Lehrer der Frau lehren geholfen, die Kühe zu melken -- nonprojective!
    ]

    terminal_rules = [
        Rule('Det_case:nom', ('der',)),
        Rule('Det_case:acc', ('den',)),
        Rule('Det_case:dat', ('dem',)),        
    ]

    rules = list(expand_gpsg_rules(
        [(rule,1) for rule in nonterminal_rules+terminal_rules],
        isem=map(str, range(num_isem)),
        tsem=map(str, range(num_tsem)),
        hsem=map(str, range(num_hsem)),
        asem=map(str, range(num_asem)),
    ))
    
    return rules

def process_pcfrs_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        new_lhs = (rule.lhs, len(rule.rhs))
        prob = monad.field.from_p(prob)
        if rule.lhs in d:
            d[new_lhs].append((rule.rhs, prob))
        else:
            d[new_lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

def make_pcfrs(monad, rules, start='S'):
    rewrites = process_pcfrs_rules(monad, rules)
    return PCFRS(monad, rewrites, start)

# PCFG : m x (a -> m [a]) x a
class PCFG(object):
    def __init__(self, monad, rewrites, start):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.join = self.monad.lift_ret(concatenate)

    # rewrite_nonterminal : a -> m [a]
    def rewrite_nonterminal(self, symbol):
        return self.rewrites[symbol]

    # is_terminal : a -> Bool
    def is_terminal(self, symbol):
        return symbol not in self.rewrites

    # rewrite_symbol : a -> m [a]
    def rewrite_symbol(self, symbol):
        if self.is_terminal(symbol):
            return self.monad.ret((symbol,))
        else:
            return self.rewrite_nonterminal(symbol) >> self.expand_string

    # expand_string : [a] x b -> m a
    def expand_string(self, string, *args):
        return self.monad.mapM(lambda s: self.rewrite_symbol(s, *args), string) >> self.join

    # distribution : m [a]
    def distribution(self):
        return self.rewrite_symbol(self.start)

    def fast_distribution(self):
        mul = self.monad.field.mul
        def gen():
            stack = [((self.start,), self.monad.field.one)]
            while stack:
                symbols, p_so_far = stack.pop()
                for i, symbol in enumerate(symbols):
                    if symbol in self.rewrites:
                        for rewrite, p in self.rewrites[symbol]:
                            rewritten = *symbols[:i], *rewrite, *symbols[i+1:]
                            stack.append((rewritten, mul(p_so_far, p)))
                        break
                else:
                    yield symbols, p_so_far
        return self.monad(gen()).marginalize().normalize()

class PCFRS(PCFG):
    # rules have format (symbol, num_blocks) -> blocks
    def expand_string(self, string, *args):
        symbols, indices = process_indexed_string(string) 
        return self.monad.mapM(
            lambda s: self.rewrite_symbol(s, *args), symbols) >> (
            lambda s: self.monad.ret(put_into_indices(s, indices)))

    def distribution(self):
        return self.rewrite_symbol((self.start, 1))    

# BoundedPCFG : m x (a -> m [a]) x a x Nat    
class BoundedPCFG(PCFG):
    """ PCFG where a symbol can only be rewritten n times recursively. 
    If symbols have indices (e.g., NP_i), the indices are ignored for the 
    purpose of counting symbols during derivations. """
    def __init__(self, monad, rewrites, start, bound):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.bound = bound
        self.join = self.monad.lift_ret(concatenate)

    def rewrite_symbol(self, symbol, history):
        if self.is_terminal(symbol):
            return self.monad.ret((symbol,))
        else:
            symbol_bare, *_ = symbol.split("_")
            condition = history.count(symbol_bare) <= self.bound
            new_history = history.add(symbol_bare)
            return self.monad.guard(condition) >> (
                lambda _: self.rewrite_nonterminal(symbol) >> (
                lambda s: self.expand_string(s, new_history)))
  
    def distribution(self):
        return self.rewrite_symbol(self.start, pyr.pbag([]))

class BoundedPCFRS(PCFRS, BoundedPCFG):
    def distribution(self):
        return self.rewrite_symbol((self.start, 1), pyr.pbag([]))

def process_indexed_string(string):
    # String is a sequence of blocks.
    # A block is a sequence of either symbols or (symbol, int, int) tuples
    # Return a tuple (symbols, indices).
    # for example if string = ( ((VP,0,0),), (V, (VP,0,1)) )
    # then symbols should be [(VP,2), (V,1)]
    # and indices contains information necesary to reconstruct the original order:
    # for each symbol, gives a sequence (b,i) of blocks and indices within those blocks.
    symbols = []
    part_of = []
    seen = {}
    for b, block in enumerate(string):
        for i, part in enumerate(block):
            if isinstance(part, str):
                symbols.append((part, 1))
                part_of.append((b,i))
            else: # TODO this is the tricky part
                symbol, index, num_blocks = part
                symbols.append((symbol)) #
                if (symbol, index) in seen:
                    part_of.append(seen[symbol, index])
                else:
                    part_of.append(i)
                    seen[symbol, index] = i
    return symbol, part_of
            
def put_into_indices(self, symbols, indices):
    # symbols is a sequence of blocks
    seen = Counter()
    def gen():
        for index in indices:
            yield symbols[index][seen[index]]
            seen[index] += 1
    return tuple(gen())

def test_pcfg():
    r1 = Rule('S', ('NP', 'VP'))
    r2 = Rule('NP', ('D', 'N'))
    r3 = Rule('VP', ('V', 'NP'))
    r4 = Rule('VP', ('V',))
    rules = [(r1, 1), (r2, 1), (r3, .25), (r4, .75)]
    pcfg = make_pcfg(Enumeration, rules)
    enum = pcfg.fast_distribution()
    assert enum.dict[('D', 'N', 'V')] == log(.75), enum.dict
    assert enum.dict[('D', 'N', 'V', 'D', 'N')] == log(.25)
    assert sum(map(exp, enum.dict.values())) == 1

def test_bounded_pcfg():
    r1 = Rule('S', ('a', 'S', 'b'))
    r2 = Rule('S', ())
    rules = [(r1, 1/2), (r2, 1/2)]
    
    pcfg = make_bounded_pcfg(Enumeration, rules, 1)
    enum = pcfg.distribution()
    assert enum.dict[('a', 'b')] == log(1/2)
    assert enum.dict[()] == log(1/2)

    pcfg = make_bounded_pcfg(Enumeration, rules, 2)
    enum = pcfg.distribution()
    assert enum.dict[()] == log(1/2)
    assert enum.dict[('a', 'b')] == log(1/4)
    assert enum.dict[('a', 'a', 'b', 'b')] == log(1/4)

    pcfg = make_bounded_pcfg(Enumeration, rules, 3)
    enum = pcfg.distribution()
    assert enum.dict[()] == log(1/2)
    assert enum.dict[('a', 'b')] == log(1/4)
    assert enum.dict[('a', 'a', 'b', 'b')] == log(1/8)
    assert enum.dict[('a', 'a', 'a', 'b', 'b', 'b')] == log(1/8)
    
def test_pcfrs():
    r1 = Rule('S', (('NP', 'VP'),))
    r2 = Rule('NP', (('D', 'N'),))
    r3 = Rule('NPR', (('D', 'N'), ('RP',)))
    r4 = Rule('VP', (('V',),))
    r5 = Rule('S', ((('NPR', 0, 0), 'VP', ('NPR', 0, 1)),))
    r6 = Rule('D', (('the',),))
    r7 = Rule('D', (('a',),))
    r8 = Rule('N', (('cat',),))
    r9 = Rule('N', (('dog',),))
    r10 = Rule('V', (('jumped',),))
    r11 = Rule('V', (('cried',),))
    r12 = Rule('RP', (('that I saw yesterday',),))
    r13 = Rule('RP', (('that belongs to Bob',),))

    rules = [
        (r1, 1), # S -> NP VP
        (r2, 1),   # NP -> D N
        (r3, 1),   # NPR -> D N, RP
        (r4, 1),   # VP -> V
        #(r5, 1/4), # S -> NPR_1 VP NPR_2
        (r6, 1/2), # D -> the
        (r7, 1/2), # D -> a
        (r8, 1/2), # N -> cat
        (r9, 1/2), # N -> dog
        (r10, 1/2), # V -> jumped
        (r11, 1/2), # V -> cried
        (r12, 1/3), # RP -> that I saw yesterday
        (r13, 2/3), # RP -> that belongs to Bob
    ]

    pcfrs = make_pcfrs(PSpaceEnumeration, rules)
    return pcfrs


if __name__ == '__main__':
    import nose
    nose.runmodule()
