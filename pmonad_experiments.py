import sys
import itertools
from collections import deque
from math import log, exp
from pathos.multiprocessing import ProcessingPool as Pool

import rfutils
import numpy as np
import scipy.special
import pandas as pd
import tqdm

import utils
import pmonad
import pcfg
import infoloc as il
import shuffles as sh
import sources as s

DELIMITER = '#'
DEFAULT_DELIMITER = utils.RightDelimiter()

def demonstrate_systematic_learning_22(num_samples=1000, num_runs=1, **kwds):
    """ 2^2!=24 possible languages. """
    # works with redundancy=2, positional=False, split_alpha=1
    # final posterior is 1/2 because of symmetry
    targets = [0, 1, 7]
    df, support, probs = learn_from_samples(K=2, V=2, num_samples=num_samples, targets=targets*num_runs, **kwds)
    grammars = list(itertools.permutations([x+'#' for x in support]))
    def gen():
        for target in targets:
            curves = il.curves_from_sequences(grammars[target], np.exp(probs))
            yield {
                'target': target,
                'grammar': grammars[target],
                'ee': il.ee(curves),
                'ms_auc': il.ms_auc(curves),
            }
    return df, probs, pd.DataFrame(gen())    
    

def demonstrate_systematic_learning_23(num_samples=20000, num_runs=1, **kwds):
    """ 2**3!=40320 possible languages. """
    targets = [
        0, # id(1,2,3)
        1, # toffoli(1,2,3) -- flip on more frequent control bits
        5040, # toffoli(not(1), not(2), 3) -- flip on less frequent control bits
        121, # cnot(2,3) -- flip on more frequent control bit
        5046, # cnot(not(2), 3) -- flip on less frequent control bit
        11536, # weakly systematic -- if positional=True, no different from strongly systematic
        10000, # nonsystematic
        15000,
        20000,
        25000,
        30000, # nonsystematic
    ]
    df, support, probs = learn_from_samples(
        K=3,
        V=2,
        num_samples=num_samples,
        targets=tqdm.tqdm(targets*num_runs),
        **kwds
    )
    grammars = list(itertools.permutations([x+'#' for x in support]))    
    def gen():
        for target in targets:
            curves = il.curves_from_sequences(grammars[target], np.exp(probs))
            yield {
                'target': target,
                'grammar': grammars[target],
                'ee': il.ee(curves),
                'ms_auc': il.ms_auc(curves),
            }
    return df, probs, pd.DataFrame(gen())

def learn_from_samples(
        targets=[0],
        K=3, # sequence length
        V=2, # vocabulary size
        p=.6,
        asymmetry=.1,
        truncate=True,
        split_alpha=.5, # sample substrings of size k with probability \propto split_alpha^k. 
        maxlen=None,
        num_samples=10000,
        redundancy=1,
        parallel=True,
        full_support=False,
        positional=False):

    """
    Language learning as cryptography. The grammar is the key to be deciphered.
    Given uniformly distribution s = f_K(m), and knowledge of p(m),
    and lots of samples of s, how long to determine K?

    Intuitively if the data distribution is flat, then the key is harder to recover,
    because there are more keys that give rise to that distribution. On the other hand,
    if the data distribution is peaky, then it is easier to recover the key,z
    because structure-preserving keys are rare. In cryptography, a good code has
    "diffusion" and "confusion", which serve to create a flat data distribution. For
    learnability, contrariwise, you want to MINIMIZE diffusion and confusion.

    If the samples to be learned from are truncated---consisting of random contiguous substrings
    from utterances---then high E should be detrimental to learning, because long-range
    correlations will be missed. Therefore, a language is more "learnable" with small samples
    if it has low E.
    """
    if maxlen is None:
        maxlen = K
    alphabet = [chr(65+v) for v in range(V)]
    if V == 2:
        probs = np.array([p, 1-p])
    else:
        raise NotImplementedError
    character_sources = [
        pmonad.Enumeration(list(zip(alphabet, np.log(probs+k*asymmetry*np.array([+1,-1])))))
        for k in range(K)
    ]
    string_source = character_sources[0].ret("")
    
    @string_source.lift_ret
    def concatenate_distinctly(c, x):
        if positional:
            return "".join([c, chr(len(c)*V + ord(x))*redundancy])
        else:
            return "".join([c, x*redundancy])
        
    for k in range(K):
        string_source = string_source >> (lambda c:
            character_sources[k] >> (lambda x:
            concatenate_distinctly(c, x)))

    source_probs = [p for _, p in string_source]
    source = pmonad.Enumeration(list(enumerate(source_probs)))
    
    if full_support:
        assert not positional
        strings = list(utils.cartesian_forms(V, K*redundancy))
    else:
        strings = [s for s, _ in string_source]
        
    grammars_support = list(itertools.permutations(strings, len(source_probs)))
    num_grammars = len(grammars_support)

    @source.lift_ret
    def produce(g, m):
        return grammars_support[g][m]

    if truncate:
        obs_support = list({
            obs
            for s in strings
            for k in range(maxlen)
            for obs in utils.padded_sliding(s, k+1)
        })
    else:
        obs_support = strings
    num_obs = len(obs_support)

    context_size = source.exponential(range(maxlen), alpha=-np.log(split_alpha))

    mul, from_p, to_log, zero = source.field.mul, source.field.from_p, source.field.to_log, source.field.zero    
    def likelihood(g):
        utterances = source >> (lambda m: produce(g, m))
        if truncate:
            # adjust probabilities
            adjusted = type(source)(
                (x, mul(from_p(len(x)+1), p))
                for x, p in utterances.values
            )
            return adjusted >> (lambda utt:
                context_size >> (lambda k:
                source.uniform(utils.padded_sliding(utt, k+1))))
        else:
            return utterances

    # arrange into joint probabilities of shape GO, evidence of shape O
    # problem: O has different support size if we are using full support
    print("Calculating likelihood array...", file=sys.stderr)
    likelihood_dicts = map(dict, map(likelihood, range(num_grammars)))
    likelihood_array = np.array([
        [
            to_log(likelihood_dict.get(obs, zero))
            for obs in obs_support
        ]
        for likelihood_dict in tqdm.tqdm(likelihood_dicts, total=num_grammars)
    ])

    def run_target(target):
        #print("Running target grammar %d" % target, file=sys.stderr)
        def gen():
            target_likelihood = np.exp(likelihood_array[target])
            log_prior_array = -log(num_grammars) * np.ones(num_grammars)
            for i in range(num_samples):
                observed = np.random.choice(
                    range(num_obs),
                    p=target_likelihood,
                )
                joint = log_prior_array + likelihood_array[:, observed] # shape G
                log_posterior_array = scipy.special.log_softmax(joint, -1) # shape G

                #marginal = log_prior_array[:, None] + likelihood_array
                
                

                yield {
                    'target': str(target),
                    'i': i,
                    'data': obs_support[observed],
                    'posterior': log_posterior_array[target],
                    'entropy': scipy.stats.entropy(np.exp(log_posterior_array)),
                    #'formal_entropy': ...,
                    #'semantic_entropy': ...,
                }
                log_prior_array = log_posterior_array
        return list(gen())

    print("Simulating learning...", file=sys.stderr)
    if parallel:
        with Pool() as p:
            results = tqdm.tqdm(p.map(run_target, targets))
    else:
        results = tqdm.tqdm(map(run_target, targets))
        
    return pd.DataFrame(utils.flat(results)), strings, source_probs
        
def head_direction_consistency(
        num_cats=5,
        num_heads=5,
        max_modifiers=1,
        with_brackets=True,
        T=1,
        bound=2,
        with_delimiter=DEFAULT_DELIMITER):
    # show: by-category consistent > consistent
    # comparing to total inconsistency is hard because it increases the entropy rate

    # need to try with arity-3 trees so that this doesn't just reduce to the regular vs. CFG comparison...
    def rules(right_branch):
        for phrase in range(num_cats):
            phrase_label = f"{phrase}P"
            for head in range(num_heads):
                yield pcfg.Rule(phrase_label, ["(", f"h{phrase}_{head}", ")"])
                for phrase2 in range(num_cats):
                    if right_branch[phrase2]:
                        yield pcfg.Rule(phrase_label, ["(", f"h{phrase}_{head}", f"{phrase2}P", ")"])
                    else:
                        yield pcfg.Rule(phrase_label, ["(", f"{phrase2}P", f"h{phrase}_{head}", ")"])

    mask = [np.random.random() < .5 for _ in range(num_cats)]
    inconsistent_rules = list(rules(mask))
    consistent_rules = list(rules([True]*num_cats))
    probs = np.exp(np.random.randn(len(consistent_rules)))
    inconsistent_grammar = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(inconsistent_rules, probs), bound, start="0P")
    consistent_grammar = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(consistent_rules, probs), bound, start="0P")
    inconsistent_forms = pcfg_forms(inconsistent_grammar, with_delimiter=with_delimiter)
    consistent_forms = pcfg_forms(consistent_grammar, with_delimiter=with_delimiter)
    return inconsistent_forms, consistent_forms, mask
                
def dutch_vs_german(
        num_n=5,
        num_v=5,
        p_embed=.5,
        independent=False,
        T=1,
        bound=3,
        with_delimiter=DEFAULT_DELIMITER):
    """ Nested vs. cross-serial dependencies, as in German vs. Dutch """
    # When independent=True, Dutch and German come out exactly the same---same dependency length effect.
    # When independent=False, Dutch is slightly better than German.
    def rules():
        for n, v in itertools.product(range(num_n), range(num_v)):
            yield pcfg.Rule("S", [f"N{n}", f"V{v}"])
            yield pcfg.Rule("S", [f"N{n}", "S", f"V{v}"])
    the_rules = list(rules())
    if not independent:
        probs = (np.exp(1/T*np.random.randn(len(the_rules) // 2, 2)) * np.array([1-p_embed, p_embed])).flatten()
    else:
        probs = (np.exp(1/T*np.random.randn(len(the_rules) // 2)[:, None]) * np.array([1-p_embed, p_embed])).flatten()
    german_grammar = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, probs), bound, start="S")
    forms = pcfg_forms(german_grammar, with_delimiter=with_delimiter)
    forms.columns = ['german', 'p']

    @utils.delimited_sequence_transformer
    def convert_to_dutch(form):
        midpoint = len(form)//2
        return form[:midpoint] + form[midpoint:][::-1]

    ds = sh.DeterministicScramble()
    
    forms['dutch'] = forms['german'].map(convert_to_dutch)
    forms['scramble'] = forms['german'].map(ds.shuffle)
    return forms

OPENERS = "([{<‘“«abcdefghijklm"
CLOSERS = ")]}>’”»zyxwvutsrqpon"

# Dyck language as head-initial dependency grammar:
# ( -> )
# ( -> [ )
# ( -> ) [
# ( -> [ ) {

def random_dyck_pcfg(num_cats=4, recurse_prob=.5, split_prob=.5, T=1, bound=2, regular=False):
    # Sx -> Sy Sz
    # Sx -> (x Sy )x
    # Sx -> (x )x
    def rules():
        for cat in range(num_cats):
            yield pcfg.Rule(f"S_{cat}", [OPENERS[cat], CLOSERS[cat]]), (1 - split_prob) * (1 - recurse_prob)
            for cat2 in range(num_cats):
                yield pcfg.Rule(f"S_{cat}", [OPENERS[cat], f"S_{cat2}", CLOSERS[cat]]), (1 - split_prob) * recurse_prob / num_cats
                for cat3 in range(num_cats):
                    yield pcfg.Rule(f"S_{cat}", [f"S_{cat2}", f"S_{cat3}"]), split_prob / num_cats**2


    the_rules, ps = zip(*rules())
    # add noise to probabilities; means that certain openers are more or less likely to split/recurse
    noisy_ps = np.exp(np.log(np.array(ps)) + 1/T*np.random.randn(len(the_rules)))
    return pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, noisy_ps), bound, start="S_0")

def random_xbar_pcfg(num_cats, num_heads, num_adjuncts, adjunct_optional=False, T=1, bound=1):
    """ Xbar-style random PCFG, with a matched finite-state grammar.
    If the PCFG is unambiguous, then its entropy is the same as its finite-state equivalent.
    """
    # The FS grammar gives better E than the PCFG, so why do human languages have PCFG?
    # Intuitively, PCFG yields better predictability within spans.
    
    # XP -> y X'
    # X' -> (ZP) x
    # where each XP is associated with disjoint sets of adjuncts y and heads x.
    def rules():
        for category in range(num_cats):
            bar_label = f"{category}'"
            for head in range(num_heads):
                head_label = f"{category}_{head}"
                yield pcfg.Rule(bar_label, [head_label]) # X' -> x                
                for subcategory in range(num_cats):
                    yield pcfg.Rule(bar_label, [f"{subcategory}P", head_label]) # X' -> ZP x
                    
            phrase_label = f"{category}P"
            if adjunct_optional:
                yield pcfg.Rule(phrase_label, [bar_label]) # XP -> X'
            for adjunct in range(num_heads):
                adjunct_label = f"a{category}_{adjunct}"
                yield pcfg.Rule(phrase_label, [adjunct_label, bar_label]) # XP -> y X'
                
    the_rules = list(rules())
    reg_rules = [ # reverse if it's a bar level rule, yielding a regular grammar
        pcfg.Rule(rule.lhs, list(reversed(rule.rhs))) if "'" in rule.lhs else rule # X' -> x ZP
        for rule in the_rules 
    ]
    probs = np.exp(1/T*np.random.randn(len(the_rules)))

    if bound is None:
        cf = pcfg.make_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, probs), start="0P")
        reg = pcfg.make_pcfg(pcfg.PSpaceEnumeration, zip(reg_rules, probs), start="0P")
    else:
        cf = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, probs), bound, start="0P")
        reg = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(reg_rules, probs), bound, start="0P")
        
    return cf, reg

def format_forms(forms, with_delimiter=DEFAULT_DELIMITER):
    def recode(x, _seen={}):
        if x in _seen:
            return _seen[x]
        else:
            _seen[x] = len(_seen)
            return _seen[x]
    def process_form(xs):
        return with_delimiter.delimit_string("".join(chr(65+recode(x)) for x in xs))
    return map(process_form, forms)

def analyze_enum(enum, with_delimiter=DEFAULT_DELIMITER, transformation=None, p_transform=1, **kwds):
    M = type(enum)
    if transformation is not None:
        if p_transform < 1:
            enum = M.flip(p_transform) >> (lambda b:
                enum >> (lambda x:
                M.ret(transformation(x) if b else x)))
        elif p_transform == 1:
            enum = enum >> M.lift_ret(transformation)
    forms, ps = zip(*enum.values)
    curves = il.curves_from_sequences(format_forms(forms, with_delimiter=with_delimiter), ps, **kwds)
    return curves

def random_cnf_pcfg(num_nt, num_t, T=1, bound=None):
    # for each nt, a distribution over nts and ts

    # NT -> NT NT, or
    # NT -> T.
    nt_grammar = np.exp(1/T*np.random.randn(num_nt, num_nt, num_nt))
    t_grammar = np.exp(1/T*np.random.randn(num_nt, num_t))
    ts = range(num_t)
    nts = range(num_nt)
    rules = [
        (pcfg.Rule(str(num_t + i), (str(j),)), t_grammar[i, j])
        for i, j in itertools.product(nts, ts)
    ]
    rules.extend(
        (pcfg.Rule(str(num_t + i), (str(num_t + j), str(num_t + k))), nt_grammar[i,j,k])
        for i, j, k in itertools.product(nts, nts, nts)
    )

    if bound is None:
        return pcfg.make_pcfg(pcfg.PSpaceEnumeration, rules, start=str(num_t))
    else:
        return pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, rules, bound, start=str(num_t))

def random_mccoy_question_grammar(
        num_nouns=13,
        num_intransitive_verbs=9,
        num_transitive_verbs=9,
        num_prepositions=8,
        num_complementizers=2,
        num_dets=6,
        num_aux=2, # positive and negative
        T=1,
        monad=pcfg.PSpaceEnumeration,
        bound=None):
    # MCFG for question formation
    # S -> NP IP
    # S -> IP_1 NP IP_2
    # IP -> I, VP
    # IP -> I VP
    def rules():
        for n in range(num_nouns):
            for n2 in range(num_nouns):
                for pl in ['sg', 'pl']:
                    for pl2 in ['sg', 'pl']:
                        for d in range(num_dets):
                            for p in range(num_prepositions):
                                for c in range(num_complementizers):
                                    for v in range(num_intransitive_verbs + num_transitive_verbs):
                                        for a in range(num_aux):
                                        
                                            yield pcfg.Rule(f"S", (f"NP_{n}_{pl}", f"VP_{v}_{pl}"))
                                            yield pcfg.Rule(f"NP_{n}_{pl}", ("Det", f"N_{n}_{pl}"))
                                            yield pcfg.Rule(f"NP_{n}_{pl}", ("Det", f"N_{n}_{pl}", f"Prep_{p}", "Det", f"N_{n2}_{pl2}"))
                                            yield pcfg.Rule(f"NP_{n}_{pl}", ("Det", f"N_{n}_{pl}", f"Rel", f"RC_{v}_{pl}"))
                                            yield pcfg.Rule("Det", (f"D_{d}",))
                                            yield pcfg.Rule("Rel", (f"C_{c}",))
                                            yield pcfg.Rule(f"Aux_{pl}", (f"Aux_{a}_{pl}",))
                                        
                                            if v < num_intransitive_verbs:
                                                yield pcfg.Rule(f"VP_{v}_{pl}", (f"Aux_{pl}", f"V_{v}",))
                                                yield pcfg.Rule(f"RC_{v}_{pl}", (f"Aux_{pl}", f"V_{v}",))
                                            else:
                                                yield pcfg.Rule(f"VP_{v}_{pl}", (f"Aux_{pl}", f"V_{v}", f"NP_{n}_{pl2}"))
                                                yield pcfg.Rule(f"RC_{v}_{pl}", (f"Aux_{pl}", f"V_{v}", "Det", f"N_{n}_{pl2}"))
                                                yield pcfg.Rule(f"RC_{v}_{pl}", ("Det", f"N_{n}_{pl2}", f"Aux_{pl2}", f"V_{v}"))

    the_rules = set(rules())
    probs = np.exp(1/T * np.random.randn(len(the_rules)))
    if bound is None:
        return pcfg.make_pcfg(pmonad.PSpaceEnumeration, zip(the_rules, probs), start="S")
    else:
        return pcfg.make_bounded_pcfg(pmonad.PSpaceEnumeration, zip(the_rules, probs), bound, start="S")
    
def random_mccoy_tense_grammar(
        num_nouns=4, #13,
        num_intransitive_verbs=4, #9,
        num_transitive_verbs=4, #9,
        num_prepositions=1, #8,
        num_complementizers=1, #2,
        num_dets=1, #6,
        T=1,
        monad=pcfg.PSpaceEnumeration,
        bound=None):
    # Lexicalized grammar based on McCoy et al. (2020, TACL)
    # Commented-out default arguments are from the original paper.

    # Seems to work with T=1/10, but not with T=1. Why?
    
    # Hypothesis 1: T=1 creates synergy among Subject:Verb:Number. T=1/10 creates redundancy.
    # It is damaging to separate synergistic bits, but good to separate redundant bits.

    # Hypothesis 2: T=1/10 creates high MI between the verb and its number.
    # Thus, plural marking on the verb does not contribute to 1-gram entropy,
    # but fake plural marking (based on an uncorrelated noun) does increase 1-gram entropy.
    # This seems to work, almost---it reduces the advantage.
    # -> Empirical claim: MI of a verb and the number of its subject > MI of a verb and the number of its nearest noun.
    # Almost certainly true! Go check.
    def rules():
        for n in range(num_nouns):
            for n2 in range(num_nouns):
                for pl in ['sg', 'pl']:
                    for pl2 in ['sg', 'pl']:
                        for d in range(num_dets):
                            for p in range(num_prepositions):
                                for c in range(num_complementizers):
                                    for v in range(num_intransitive_verbs + num_transitive_verbs):
                                        
                                        yield pcfg.Rule(f"S", (f"NP_{n}_{pl}", f"VP_{v}_{pl}"))
                                        yield pcfg.Rule(f"NP_{n}_{pl}", ("Det", f"N_{n}_{pl}"))
                                        yield pcfg.Rule(f"NP_{n}_{pl}", ("Det", f"N_{n}_{pl}", f"Prep_{p}", "Det", f"N_{n2}_{pl2}"))
                                        yield pcfg.Rule(f"NP_{n}_{pl}", ("Det", f"N_{n}_{pl}", f"Rel", f"RC_{v}_{pl}"))
                                        yield pcfg.Rule("Det", (f"D_{d}",))
                                        yield pcfg.Rule("Rel", (f"C_{c}",))
                                
                                        if v < num_intransitive_verbs:
                                            yield pcfg.Rule(f"VP_{v}_{pl}", (f"V_{v}_{pl}",))
                                            yield pcfg.Rule(f"RC_{v}_{pl}", (f"V_{v}_{pl}",))
                                        else:
                                            yield pcfg.Rule(f"VP_{v}_{pl}", (f"V_{v}_{pl}", f"NP_{n}_{pl2}"))
                                            yield pcfg.Rule(f"RC_{v}_{pl}", (f"V_{v}_{pl}", "Det", f"N_{n}_{pl2}"))
                                            yield pcfg.Rule(f"RC_{v}_{pl}", ("Det", f"N_{n}_{pl2}", f"V_{v}_{pl2}"))

    the_rules = set(rules())
    probs = np.exp(1/T * np.random.randn(len(the_rules)))
    sv_probs = np.array(list(dict(sorted((r,p) for r, p in zip(the_rules, probs) if r.lhs == 'S')).values())).reshape(num_nouns, 2, num_intransitive_verbs + num_transitive_verbs)
    sv_probs /= sv_probs.sum()
    
    if bound is None:
        return pcfg.make_pcfg(pmonad.PSpaceEnumeration, zip(the_rules, probs), start="S"), s.coinformation_lattice(sv_probs)
    else:
        return pcfg.make_bounded_pcfg(pmonad.PSpaceEnumeration, zip(the_rules, probs), bound, start="S"), s.coinformation_lattice(sv_probs)

def move_matrix_aux(terminals):
    def gen():
        embedded = False
        found_aux = False
        so_far = []
        for terminal in terminals:
            if not found_aux and not embedded and terminal.startswith("Aux"):
                found_aux = True
                yield terminal
                yield from so_far
            elif not found_aux and embedded and terminal.startswith("Aux"): # only one level of embedding possible under the grammar
                embedded = False
                so_far.append(terminal)
            elif not found_aux and terminal.startswith("C"):
                embedded = True
                so_far.append(terminal)
            elif not found_aux:
                so_far.append(terminal)
            else:
                yield terminal
    return tuple(gen())

def move_first_aux(terminals):
    def gen():
        found_aux = False
        so_far = []
        for terminal in terminals:
            if not found_aux and terminal.startswith("Aux"):
                found_aux = True
                yield terminal
                yield from so_far
            elif not found_aux:
                so_far.append(terminal)
            else:
                yield terminal
    return tuple(gen())
    
def change_agreement(terminals):
    # ('Det', 'N_2_pl', 'Prep', 'Det', 'N_2_pl', 'V_5_pl', 'Det', 'N_1_pl', 'Rel', 'Det', 'N_2_sg', 'V_5_sg')
    # the doctors in the houses hate the monkeys that the criminal hates
    # modify now so that every V_{v}_{pl} agrees with the *most recent* noun.
    def gen():
        plural_state = None
        for terminal in terminals:
            if terminal.startswith("N"):
                _, _, plural_state = terminal.split("_")
                yield terminal
            elif terminal.startswith("V"):
                V, v, _ = terminal.split("_")
                yield "_".join([V, v, plural_state])
            else:
                yield terminal
    return tuple(gen())

def rearrange_string_fifo(string, opening_symbols=OPENERS, closing_symbols=CLOSERS):
    # Mapping from opening to closing symbols for quick lookup and queue operations
    opening_to_closing = {opening: closing for opening, closing in zip(opening_symbols, closing_symbols)}

    def gen():
        # Queue to track the expected order of closing symbols according to FIFO logic
        queue = deque()
        for char in string:
            if char in opening_symbols:
                yield char
                queue.append(opening_to_closing[char])
            elif char in closing_symbols:
                yield queue.popleft()
            else:
                raise ValueError("Unrecognized symbol encountered.")
        if queue:
            raise ValueError("Queue not empty at the end. Unmatched opening symbols remain.")
    
    return ''.join(gen())


# Structure-sensitivity experiments
# - McCoy & Linzen examples
# - Dyck vs. scramble-Dyck vs. FIFO-Dyck
# 

# What do the structure-sensitivity experiments need?
# - Show structure-sensitive is better than (1) weak baselines, (2) strong targeted baselines.
# - Show structure-sensitivity is *THE BEST*, of all reorderings.
# - Show when nested structures are favored over regular structures.


# Dyck variants:
# ()[]() -- no nesting. But then not as many strings per length. Cannot transform from CFG while preserving entropy rate.
# ([{)]} -- cross-serial. Preserves bracket dependency length. But gets bad when there is multivariate MI. Check this.

# ([{)]} seems counterintuitive, but abcxyz seems intuitive.

 # Dyck parameters that matter:
 # - split prob / arity.
 # - recurse prob. / depth.
 # - MI of ( with ) in ().   (dependency MI)
 # - MI of ( with [ in ([]). (recurse MI)
 # - MI of ( with [ in ()[]. (split MI)

 # These are just too slow!





