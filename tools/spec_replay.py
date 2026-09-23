#!/usr/bin/env python
"""
spec_replay.py - teacher-forced replay of the --spec drafting policies.

WHAT IT MEASURES
    `--spec N` proposes up to N tokens by n-gram lookup over the sequence so far and
    verifies them in one batched greedy pass. A step pays only when enough of the
    proposal is accepted. This tool replays real text as if it were the model's greedy
    output, runs each drafting policy at every step exactly where the engine would run
    it, and records how often a draft fires, how long it is and how much of it the text
    confirms. Those counts go through the verify-step cost model below to give expected
    speedups, with bootstrap confidence intervals over documents.

WHAT IT IS NOT
    A measurement of K3. Two substitutions stand between these numbers and the engine:
      1. the "output" is human-written text, not K3's greedy continuation of the prompt;
      2. the ids come from a byte-level BPE trained here on text disjoint from the
         evaluation text, because K3's own tokenizer ships with the checkpoint.
    "Accepted" therefore means "equal to the next ids of the reference text". A model
    copies its context at a different rate than a human author does, and a different
    vocabulary moves n-gram statistics. Use the numbers to rank policies and to size the
    effect, not as a tokens-per-second forecast.

POLICIES (every one reads only the ids before the draft point)
    P0  spec_draft() in src/cli/k3_run.c, exactly: suffix 4-gram then 3-gram, the two
        most recent occurrences, the draft stops where their continuations disagree or
        where the older one runs into the newer one, K3_SPEC_MAX 8. `spec_draft` below
        is a line-for-line port and `NgramIndex` computes the same drafts incrementally;
        the tests hold the two equal.
    P1  eager most-recent match: suffix 4-gram, then 3, then 2; copy what followed the
        most recent occurrence, up to the end of the history; no agreement gate.
    P2  longest suffix match over the whole history (online suffix automaton), most
        recent occurrence of that suffix, fires when the match is at least `min_len`
        ids; the copy continues periodically past the end of the history (an LZ77-style
        overlapping copy), so a run keeps drafting.
    P3  P2 with an adaptive length: draft token j while the estimated probability that
        tokens 1..j are ALL accepted exceeds delta, the marginal cost of verifying one
        more position (0.22 at 8 GB, 0.55 at 128 GB). The estimate is a table over
        (match length + j - 1, occurrence and agreement counts of the matched context,
        depth), fitted on calibration documents disjoint from the evaluation documents.

COST MODEL (units of one plain decode token)
    A verify step over n = nd + 1 positions costs S(n) = a_T + a_E*u(n) + a_C*n + eps,
    u(n) = 56*(1 - (55/56)^n), the expected distinct-expert factor at 16 of 896 experts.
      8 GB   (trunk streamed):  a_T 0.78  a_E 0.19  a_C 0.03
      128 GB (trunk resident):  a_T 0.44  a_E 0.45  a_C 0.11
    A step with no draft costs 1 and emits 1. Today a partial acceptance (m < nd) pays a
    second pass of a_T + a_E*u(m+1) + a_C*(m+1) to restore and replay the accepted
    prefix; the replay-free rollback removes that pass. A step emits m + 1 ids (the m
    accepted drafts plus the model's own next id), truncated at the end of the turn.

usage:
    spec_replay.py run --work DIR [--out docs/measurements/spec-replay.json]
        build the corpora, train the proxy tokenizer (cached in DIR), replay, write JSON
    spec_replay.py replay-ids IDS.jsonl [--out OUT.json]
        replay the headline policies on ids the engine emitted, using the P3 table and
        P2 defaults recorded in docs/measurements/spec-replay.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

K3_SPEC_MAX = 8

TIERS = {
    "8GB": {"a_T": 0.78, "a_E": 0.19, "a_C": 0.03},
    "128GB": {"a_T": 0.44, "a_E": 0.45, "a_C": 0.11},
}
DELTA = {"8GB": 0.22, "128GB": 0.55}


# ---------------------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------------------

def distinct_experts(n):
    """Expected distinct experts over n positions, in units of one position's 16."""
    return 56.0 * (1.0 - (55.0 / 56.0) ** n)


def verify_cost(nd, tier, eps=0.0):
    """Cost of one decode step that verifies nd drafts; nd == 0 is a plain step."""
    if nd <= 0:
        return 1.0
    c = TIERS[tier]
    n = nd + 1
    return c["a_T"] + c["a_E"] * distinct_experts(n) + c["a_C"] * n + eps


def replay_cost(m, tier):
    """Today's second pass after a partial acceptance: restore, replay m + 1 positions."""
    c = TIERS[tier]
    n = m + 1
    return c["a_T"] + c["a_E"] * distinct_experts(n) + c["a_C"] * n


def step_cost(nd, m, tier, replay, eps=0.0):
    cost = verify_cost(nd, tier, eps)
    if replay and 0 <= m < nd:
        cost += replay_cost(m, tier)
    return cost


def marginal_cost(j, tier, eps=0.0):
    """Replay-free cost of verifying draft j (1-based) on top of drafts 1..j-1."""
    return verify_cost(j, tier, eps) - verify_cost(j - 1, tier, eps)


# ---------------------------------------------------------------------------------------
# P0 and P1: ports and their incremental equivalents
# ---------------------------------------------------------------------------------------

def spec_draft(seq, T, cap):
    """Line-for-line port of spec_draft() in src/cli/k3_run.c; returns the draft ids.

    seq[0..T-1] is the sequence so far (prompt + generated); the draft proposes seq[T..].
    The C for-loops are written as while-loops so that `continue` keeps the C semantics
    of running the loop increment.
    """
    out = [0] * K3_SPEC_MAX
    if cap > K3_SPEC_MAX:
        cap = K3_SPEC_MAX
    n = 4
    while n >= 3:
        if T < n + 1:
            n -= 1
            continue
        m1, m2 = -1, -1                                  # two most recent matches
        j = T - n - 1
        while j >= 0:
            hit = 1
            for i in range(n):
                if seq[j + i] != seq[T - n + i]:
                    hit = 0
                    break
            if not hit:
                j -= 1
                continue
            if m1 < 0:
                m1 = j
            else:
                m2 = j
                break
            j -= 1
        if m1 < 0:
            n -= 1
            continue
        nd = 0
        i = 0
        while nd < cap and m1 + n + i < T:
            cand = seq[m1 + n + i]
            if m2 >= 0:
                # stop where the two histories stop agreeing
                if m2 + n + i >= m1 or seq[m2 + n + i] != cand:
                    break
            out[nd] = cand
            nd += 1
            i += 1
        if nd > 0:
            return out[:nd]
        n -= 1
    return []


def eager_draft(seq, T, cap):
    """P1 by backward scan: most recent suffix 4-gram, then 3, then 2; no agreement gate."""
    if cap > K3_SPEC_MAX:
        cap = K3_SPEC_MAX
    for n in (4, 3, 2):
        if T < n + 1:
            continue
        for j in range(T - n - 1, -1, -1):
            if all(seq[j + i] == seq[T - n + i] for i in range(n)):
                return [seq[j + n + i] for i in range(cap) if j + n + i < T]
    return []


class NgramIndex:
    """Incremental form of the backward scans in spec_draft() and eager_draft().

    After advance() has been called T times, every start j <= T-n-1 of an n-gram of
    seq[0..T-1] is indexed, in increasing order. That is exactly the range the C loop
    scans: it starts at j = T-n-1 so the suffix never matches itself, and the most recent
    match is the largest such j.
    """

    ORDERS = (2, 3, 4)

    def __init__(self, seq):
        self.seq = seq
        self.T = 0
        self.ix = {n: {} for n in self.ORDERS}

    def advance(self):
        seq = self.seq
        self.T += 1
        T = self.T
        for n, ix in self.ix.items():
            j = T - n - 1
            if j >= 0:
                key = tuple(seq[j:j + n])
                lst = ix.get(key)
                if lst is None:
                    ix[key] = [j]
                else:
                    lst.append(j)

    def _recent(self, n):
        T = self.T
        if T < n + 1:
            return None
        return self.ix[n].get(tuple(self.seq[T - n:T]))

    def p0(self, cap):
        seq, T = self.seq, self.T
        cap = min(cap, K3_SPEC_MAX)
        for n in (4, 3):
            lst = self._recent(n)
            if not lst:
                continue
            m1 = lst[-1]
            m2 = lst[-2] if len(lst) > 1 else -1
            out = []
            i = 0
            while len(out) < cap and m1 + n + i < T:
                cand = seq[m1 + n + i]
                if m2 >= 0 and (m2 + n + i >= m1 or seq[m2 + n + i] != cand):
                    break
                out.append(cand)
                i += 1
            if out:
                return out
        return []

    def p1(self, cap):
        seq, T = self.seq, self.T
        cap = min(cap, K3_SPEC_MAX)
        for n in (4, 3, 2):
            lst = self._recent(n)
            if lst:
                m1 = lst[-1]
                return seq[m1 + n:min(m1 + n + cap, T)]
        return []


# ---------------------------------------------------------------------------------------
# P2: online suffix automaton with occurrence bookkeeping
# ---------------------------------------------------------------------------------------

class SuffixAutomaton:
    """Suffix automaton over token ids, extended one id at a time.

    Per state it keeps last_end, the largest end position of any occurrence, and cnt,
    the number of occurrences. Both are updated along the suffix-link path in commit(),
    which the caller runs AFTER reading the state returned by extend(): until then that
    state's figures exclude the id just appended, i.e. they describe the history before
    it, which is what a drafter needs.
    """

    def __init__(self):
        self.nxt = [{}]
        self.link = [-1]
        self.length = [0]
        self.last_end = [-1]
        self.cnt = [0]
        self.last = 0

    def extend(self, c, pos):
        """Append id c at position pos. Returns v, the state of the longest suffix of
        seq[0..pos] that also ends somewhere before pos (0, the root, when none does)."""
        nxt, link, length = self.nxt, self.link, self.length
        cur = len(length)
        length.append(length[self.last] + 1)
        link.append(0)
        nxt.append({})
        self.last_end.append(pos)
        self.cnt.append(1)
        p = self.last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur
            p = link[p]
        if p != -1:
            q = nxt[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                nxt.append(nxt[q].copy())
                self.last_end.append(self.last_end[q])
                self.cnt.append(self.cnt[q])
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self.last = cur
        return link[cur]

    def commit(self, pos):
        link, last_end, cnt = self.link, self.last_end, self.cnt
        v = link[self.last]
        while v > 0:
            last_end[v] = pos
            cnt[v] += 1
            v = link[v]


def longest_suffix_match(seq, T):
    """Brute-force reference for P2: (L, e, count) for the longest suffix of seq[0..T-1]
    that also ends at some e <= T-2, e its most recent such end, count its occurrences
    ending at <= T-2. (0, -1, 0) when even the last id never occurred before."""
    best = (0, -1, 0)
    for L in range(1, T):
        suf = seq[T - L:T]
        ends = [e for e in range(L - 1, T - 1) if seq[e - L + 1:e + 1] == suf]
        if not ends:
            break
        best = (L, max(ends), len(ends))
    return best


def periodic_copy(seq, start, g, n):
    """seq[start..] continued with period g past the end of the history."""
    return [seq[start + (i % g)] for i in range(n)]


# ---------------------------------------------------------------------------------------
# P3 features: effective match length, occurrence counts, depth
# ---------------------------------------------------------------------------------------

_ELL_EDGES = (1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 24, 32, 64)
N_ELL = len(_ELL_EDGES)
_ELL_LUT = [0] * 65
for _b, _lo in enumerate(_ELL_EDGES):
    for _v in range(_lo, 65):
        _ELL_LUT[_v] = _b
N_AGREE = 6
N_DEPTH = 3
N_CODES = N_DEPTH * N_ELL * N_AGREE
NO_DRAFT = 255
AGREE_NAMES = ("never followed", "one occurrence", "unanimous 2-3", "unanimous 4+",
               "majority", "minority")


def ell_bucket(ell):
    return _ELL_LUT[ell] if ell < 64 else N_ELL - 1


def agree_bucket(n, k):
    """n prior occurrences of the context, k of them followed by the proposed id."""
    if k <= 0:
        return 0
    if n <= 1:
        return 1
    if k >= n:
        return 2 if n <= 3 else 3
    return 4 if 2 * k >= n else 5


def depth_bucket(j):
    return 0 if j == 1 else (1 if j <= 3 else 2)


def feature_code(j, ell, n, k):
    return (depth_bucket(j) * N_ELL + ell_bucket(ell)) * N_AGREE + agree_bucket(n, k)


def draft_features(sam, v, L, d):
    """Feature code per draft position: context = matched suffix + d[:j-1]."""
    nxt, cnt = sam.nxt, sam.cnt
    codes = []
    w = v
    for j in range(len(d)):
        if w >= 0:
            n = cnt[w]
            w2 = nxt[w].get(d[j], -1)
            k = cnt[w2] if w2 >= 0 else 0
        else:
            n, k, w2 = 0, 0, -1
        codes.append(feature_code(j + 1, L + j, n, k))
        w = w2
    return codes


# ---------------------------------------------------------------------------------------
# one pass over a document: every policy's draft at every draft point
# ---------------------------------------------------------------------------------------

def _lcp(d, seq, T, r):
    m = 0
    lim = min(len(d), r)
    while m < lim and seq[T + m] == d[m]:
        m += 1
    return m


def analyze_sequence(seq, segments):
    """Drafts and acceptances of every policy at every draft point of one document.

    seq       all ids of the conversation, prompts and outputs
    segments  (start, end) ranges of model output; seq[end-1] is the turn's end id.
              Draft points are T in [start+1, end): the first id of a turn comes out of
              the prefill pass, where the engine never drafts.
    For each draft point the arrays hold: rem, ids left in the turn (end - T); l0/a0 and
    l1/a1, P0's and P1's draft length and accepted prefix (drafts at the 8-id cap; a
    shorter cap N accepts min(a, N)); L2, P2's match length (0 = none); g2, history ids
    after P2's occurrence (the no-overlap draft length is min(g2, 8)); a2, accepted prefix
    of P2's 8-id draft; f3, P3 feature codes for draft positions 1..8.
    """
    n = len(seq)
    pred_end = [0] * (n + 1)
    for s, e in segments:
        for T in range(s + 1, e):
            pred_end[T] = e
    rem, l0, a0, l1, a1, L2, g2, a2, f3 = ([] for _ in range(9))
    idx = NgramIndex(seq)
    sam = SuffixAutomaton()
    length, last_end = sam.length, sam.last_end
    no_feat = [NO_DRAFT] * K3_SPEC_MAX
    for pos in range(n):
        idx.advance()
        v = sam.extend(seq[pos], pos)
        T = pos + 1
        e = pred_end[T]
        if e:
            r = e - T
            rem.append(r)
            d = idx.p0(K3_SPEC_MAX)
            l0.append(len(d))
            a0.append(_lcp(d, seq, T, r))
            d = idx.p1(K3_SPEC_MAX)
            l1.append(len(d))
            a1.append(_lcp(d, seq, T, r))
            if v > 0:
                L = length[v]
                g = pos - last_end[v]
                d = periodic_copy(seq, last_end[v] + 1, g, K3_SPEC_MAX)
                L2.append(L)
                g2.append(g)
                a2.append(_lcp(d, seq, T, r))
                f3.extend(draft_features(sam, v, L, d))
            else:
                L2.append(0)
                g2.append(0)
                a2.append(0)
                f3.extend(no_feat)
        sam.commit(pos)
    small = np.int16
    return {
        "rem": np.asarray(rem, np.int32),
        "l0": np.asarray(l0, small), "a0": np.asarray(a0, small),
        "l1": np.asarray(l1, small), "a1": np.asarray(a1, small),
        "L2": np.asarray(L2, np.int32), "g2": np.asarray(g2, np.int32),
        "a2": np.asarray(a2, small),
        "f3": np.asarray(f3, np.uint8).reshape(-1, K3_SPEC_MAX),
    }


# ---------------------------------------------------------------------------------------
# P3: acceptance table and adaptive length
# ---------------------------------------------------------------------------------------

def fit_acceptance_table(f3, acc, rem, weights=None, alpha=20.0):
    """Per-code probability that draft position j is accepted given 1..j-1 were.

    A draft point contributes a sample at depth j when its first j-1 draft ids were
    accepted and j is still inside the turn; the outcome is acc >= j. Sparse codes back
    off to the same (length, agreement) cell over all depths, then to the length alone,
    then to the global rate, each with `alpha` pseudo-samples.
    """
    if weights is None:
        weights = np.ones(len(acc))
    acc = acc.astype(np.int64)
    rem = rem.astype(np.int64)
    has = f3[:, 0] != NO_DRAFT
    hits = np.zeros(N_CODES)
    trials = np.zeros(N_CODES)
    for j in range(1, K3_SPEC_MAX + 1):
        mask = has & (acc + 1 >= j) & (rem >= j)
        c = f3[mask, j - 1].astype(np.int64)
        w = weights[mask]
        y = acc[mask] >= j
        trials += np.bincount(c, w, minlength=N_CODES)
        hits += np.bincount(c[y], w[y], minlength=N_CODES)
    ht = hits.reshape(N_DEPTH, N_ELL, N_AGREE)
    tt = trials.reshape(N_DEPTH, N_ELL, N_AGREE)
    root = hits.sum() / max(trials.sum(), 1e-9)
    q_ell = (ht.sum(axis=(0, 2)) + alpha * root) / (tt.sum(axis=(0, 2)) + alpha)
    q_cell = (ht.sum(axis=0) + alpha * q_ell[:, None]) / (tt.sum(axis=0) + alpha)
    q = (ht + alpha * q_cell[None, :, :]) / (tt + alpha)
    return q.reshape(-1), {"root": float(root), "samples": float(trials.sum()),
                           "trials": trials, "hits": hits}


def p3_calibration(f3, acc, q, delta):
    """Does the table's estimate match what happens? Over every draft point where P3
    would fire: predicted vs realized acceptance of the first draft id, and predicted
    (sum of the cumulative estimates) vs realized accepted ids per firing."""
    lens = adaptive_lengths(f3, q, delta)
    fire = lens > 0
    if not fire.any():
        return None
    P = np.cumprod(q[np.where(f3 == NO_DRAFT, 0, f3).astype(np.int64)], axis=1)
    within = np.arange(K3_SPEC_MAX)[None, :] < lens[:, None]
    acc = acc.astype(np.int64)
    return {"points_firing": round(float(fire.mean()), 4),
            "predicted_first": round(float(P[fire, 0].mean()), 4),
            "realized_first": round(float((acc[fire] >= 1).mean()), 4),
            "predicted_accepted_per_fire": round(float((P * within)[fire].sum(1).mean()), 4),
            "realized_accepted_per_fire": round(float(np.minimum(acc, lens)[fire].mean()),
                                                4)}


def adaptive_lengths(f3, q, delta, cumulative=True):
    """P3's draft length: leading positions whose estimated probability of being
    accepted (all of 1..j when cumulative, else position j alone) exceeds delta."""
    has = f3[:, 0] != NO_DRAFT
    qq = q[np.where(f3 == NO_DRAFT, 0, f3).astype(np.int64)]
    if cumulative:
        ok = np.cumprod(qq, axis=1) > delta
    else:
        ok = np.cumprod(qq > delta, axis=1).astype(bool)
    lens = ok.sum(axis=1)
    lens[~has] = 0
    return lens


def cost_aware_lengths(f3, q, tier, replay):
    """P3 with the stopping rule read off the cost model instead of a fixed delta.

    With P_j the estimated probability that drafts 1..j are all accepted (P_0 = 1),
    adding draft j gains P_j expected ids and costs the marginal verify cost
    S(j+1) - S(j); with today's replay it also turns the event "1..j-1 accepted, j
    rejected", probability P_{j-1} - P_j, into a replay of j positions. Draft j is added
    while P_j > S(j+1) - S(j) + [replay] (P_{j-1} - P_j) * R(j-1).
    """
    has = f3[:, 0] != NO_DRAFT
    P = np.cumprod(q[np.where(f3 == NO_DRAFT, 0, f3).astype(np.int64)], axis=1)
    extra = np.array([marginal_cost(j, tier) for j in range(1, K3_SPEC_MAX + 1)])[None, :]
    if replay:
        prev = np.concatenate([np.ones((len(P), 1)), P[:, :-1]], axis=1)
        R = np.array([replay_cost(j - 1, tier) for j in range(1, K3_SPEC_MAX + 1)])
        extra = extra + (prev - P) * R[None, :]
    lens = np.cumprod(P > extra, axis=1).sum(axis=1)
    lens[~has] = 0
    return lens


# ---------------------------------------------------------------------------------------
# step simulation and per-document sums
# ---------------------------------------------------------------------------------------

STAT_COLS = (["steps", "fired", "drafted", "accepted", "emitted",
              "cost_8GB_free", "cost_8GB_today", "cost_128GB_free", "cost_128GB_today"]
             + ["att%d" % i for i in range(1, K3_SPEC_MAX + 1)]
             + ["acc%d" % i for i in range(1, K3_SPEC_MAX + 1)])
COL = {name: i for i, name in enumerate(STAT_COLS)}


def _cost_tables():
    vc = {t: np.array([verify_cost(k, t) for k in range(K3_SPEC_MAX + 1)]) for t in TIERS}
    rc = {t: np.array([replay_cost(k, t) for k in range(K3_SPEC_MAX + 1)]) for t in TIERS}
    return vc, rc


_VC, _RC = _cost_tables()


def orbit(emit):
    """Draft points the engine actually visits: from each, it jumps by what it emitted."""
    t, n, out = 0, len(emit), []
    app = out.append
    while t < n:
        app(t)
        t += emit[t]
    return out


def simulate(nd, acc, rem, doc, n_docs):
    """Replay one policy configuration over concatenated draft points.

    nd   ids drafted at each point (0 = no draft), acc accepted prefix of the full draft,
    rem  ids left in the turn, doc document index. Returns per-document sums, STAT_COLS.
    """
    nd = np.asarray(nd, np.int64)
    m = np.minimum(np.asarray(acc, np.int64), nd)
    emit = np.minimum(m + 1, np.asarray(rem, np.int64))
    vis = np.asarray(orbit(emit.tolist()), np.int64)
    ndv, mv, ev, dv = nd[vis], m[vis], emit[vis], doc[vis]
    cols = [np.ones(len(vis)), (ndv > 0).astype(float), ndv, mv, ev]
    partial = mv < ndv
    for tier in TIERS:
        free = _VC[tier][ndv]
        cols.append(free)
        cols.append(free + partial * _RC[tier][mv])
    for i in range(1, K3_SPEC_MAX + 1):
        cols.append(((ndv >= i) & (mv >= i - 1)).astype(float))
    for i in range(1, K3_SPEC_MAX + 1):
        cols.append((mv >= i).astype(float))
    X = np.zeros((n_docs, len(STAT_COLS)))
    for c, arr in enumerate(cols):
        X[:, c] = np.bincount(dv, weights=np.asarray(arr, float), minlength=n_docs)
    return X


def metrics(S, eps=0.0):
    """Ratios from summed STAT_COLS (last axis); works on one row or a bootstrap stack."""
    g = lambda name: S[..., COL[name]]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "fire_rate": g("fired") / g("steps"),
            "drafted_per_fire": g("drafted") / g("fired"),
            "accepted_per_fire": g("accepted") / g("fired"),
            "acceptance_rate": g("accepted") / g("drafted"),
            "tokens_per_step": g("emitted") / g("steps"),
            "accepted_share": g("accepted") / g("emitted"),
        }
        for tier in TIERS:
            for mode in ("free", "today"):
                cost = g("cost_%s_%s" % (tier, mode)) + eps * g("fired")
                out["speedup_%s_%s" % (tier, mode)] = g("emitted") / cost
        for i in range(1, K3_SPEC_MAX + 1):
            out["pos%d" % i] = g("acc%d" % i) / g("att%d" % i)
    return out


def bootstrap_weights(n_docs, B, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=B).astype(float)


def summarize(X, W, eps=0.0):
    """Point estimate (pooled over documents) and 95% percentile interval per metric."""
    point = metrics(X.sum(axis=0), eps)
    boot = metrics(W @ X, eps)
    out = {}
    for k, v in point.items():
        b = boot[k]
        b = b[np.isfinite(b)]
        if not np.isfinite(v) or len(b) == 0:
            out[k] = None
        else:
            lo, hi = np.percentile(b, [2.5, 97.5])
            out[k] = [round(float(v), 4), round(float(lo), 4), round(float(hi), 4)]
    return out


# ---------------------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------------------

def _sha(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


def _read(path, limit=None):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if limit is not None and len(data) > limit:
        return None
    return data.decode("utf-8", "replace")


def _cut_third(text):
    """Split at the first line boundary after a third of the characters."""
    cut = text.find("\n", len(text) // 3)
    if cut < 0:
        return None
    return text[:cut + 1], text[cut + 1:]


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True
                          ).stdout


def code_edit_docs(repo, rev, min_bytes=512, max_bytes=65536):
    """(a) old file + one instruction line -> new file, for every .c/.h/.py/.md
    modification in the non-merge history reachable from rev."""
    log = _git(repo, "log", "--no-merges", "--diff-filter=M", "--name-status",
               "--format=@@%H%x09%s", rev).decode("utf-8", "replace")
    docs = []
    commit = subject = None
    for line in log.splitlines():
        if line.startswith("@@"):
            commit, _, subject = line[2:].partition("\t")
            continue
        if not line.startswith("M\t"):
            continue
        path = line[2:]
        if not path.endswith((".c", ".h", ".py", ".md")):
            continue
        try:
            old = _git(repo, "show", "%s^:%s" % (commit, path))
            new = _git(repo, "show", "%s:%s" % (commit, path))
        except subprocess.CalledProcessError:
            continue
        if old == new or not (min_bytes <= len(old) <= max_bytes):
            continue
        if not (min_bytes <= len(new) <= max_bytes):
            continue
        old_t = old.decode("utf-8", "replace")
        instr = ("Apply this change to %s and reply with the complete new file: %s\n"
                 % (path, subject.strip()))
        prompt = old_t + ("" if old_t.endswith("\n") else "\n") + "\n" + instr
        docs.append({"id": "edit:%s:%s" % (commit[:12], path), "corpus": "code_edits",
                     "turns": [(prompt, new.decode("utf-8", "replace"))]})
    return docs


def code_writing_docs(root, n_max, min_bytes=3000, max_bytes=40000):
    """(b) first third of a Python standard-library module -> the rest of it."""
    skip = ("/test/", "/tests/", "/idle_test/", "site-packages", "dist-packages",
            "__pycache__")
    cands = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if not fn.endswith(".py") or any(s in p + "/" for s in skip):
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if min_bytes <= size <= max_bytes:
                cands.append(os.path.relpath(p, root))
    cands.sort(key=_sha)
    docs = []
    for rel in cands:
        if len(docs) >= n_max:
            break
        text = _read(os.path.join(root, rel))
        split = _cut_third(text) if text else None
        if split:
            docs.append({"id": "stdlib:" + rel, "corpus": "code_writing", "turns": [split]})
    return docs


def _md_key(root_kind, rel):
    """Identity of a markdown file across installs, so every copy of one package's
    docs lands on the same side of the tokenizer/evaluation split."""
    parts = rel.split("/")
    if root_kind == "node":
        if "node_modules" in parts:
            parts = parts[len(parts) - 1 - parts[::-1].index("node_modules") + 1:]
        pkg = "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]
    elif root_kind == "ruby":
        if "gems" in parts:
            parts = parts[len(parts) - 1 - parts[::-1].index("gems") + 1:]
        pkg = parts[0].rsplit("-", 1)[0] if "-" in parts[0] else parts[0]
        parts = [pkg] + parts[1:]
    else:
        pkg = root_kind
    return pkg, root_kind + ":" + "/".join(parts)


def markdown_pool(roots, min_bytes, max_bytes):
    """{key: (package, text)} for markdown under roots = [(kind, dir)], deduplicated."""
    pool = {}
    for kind, root in roots:
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in sorted(fns):
                if not fn.lower().endswith(".md"):
                    continue
                p = os.path.join(dp, fn)
                try:
                    size = os.path.getsize(p)
                except OSError:
                    continue
                if not (min_bytes <= size <= max_bytes):
                    continue
                pkg, key = _md_key(kind, os.path.relpath(p, root))
                if key in pool:
                    continue
                text = _read(p)
                if text:
                    pool[key] = (kind + ":" + pkg, text)
    return pool


def _sections(text):
    """(heading, body) for each '#'-heading of a markdown document, plus the intro."""
    out, head, body, fence = [], None, [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and line.startswith("#"):
            out.append((head, "".join(body)))
            head, body = line.strip("# \n\t") or "this section", []
        else:
            body.append(line)
    out.append((head, "".join(body)))
    return out


def chat_turns(title, text, max_turns=10, min_body=200):
    """A technical Q&A conversation synthesized from a sectioned document: the user asks
    about each heading in turn, the assistant answers with that section's text."""
    turns = []
    for head, body in _sections(text):
        if len(body.strip()) < min_body:
            continue
        q = ("I'm reading about %s. Can you give me an overview?" % title if head is None
             else "Can you explain \"%s\"?" % head)
        turns.append((q + "\n", body.strip("\n") + "\n"))
        if len(turns) >= max_turns:
            break
    return turns


def is_calibration(doc_id):
    return int(_sha("calib:" + doc_id)[:8], 16) % 5 == 0


def build_corpora(args):
    """Evaluation documents for the four corpora, and the markdown left for training."""
    corpora = {}
    corpora["code_edits"] = code_edit_docs(args.repo, args.rev)
    corpora["code_writing"] = code_writing_docs(args.stdlib, args.n_writing)

    # prose and chat: this repository's docs, plus half of the installed packages'
    # markdown; the other half of the packages feeds the tokenizer.
    roots = [("node", r) for r in args.node_roots] + [("ruby", r) for r in args.ruby_roots]
    roots += [("go", r) for r in args.go_roots]
    pool = markdown_pool(roots, 2000, 60000)
    eval_pool, train_md = {}, []
    for key, (pkg, text) in sorted(pool.items()):
        if int(_sha("tok:" + pkg)[:8], 16) % 5 < 2:
            train_md.append(text)
        else:
            eval_pool[key] = text
    repo_md = []
    for path in _git(args.repo, "ls-tree", "-r", "--name-only", args.rev).decode().split():
        if path.endswith(".md"):
            blob = _git(args.repo, "show", "%s:%s" % (args.rev, path))
            if 2000 <= len(blob) <= 60000:
                repo_md.append(("repo:" + path, blob.decode("utf-8", "replace")))
    prose, chat = [], []
    for key, text in repo_md:
        split = _cut_third(text)
        if split:
            prose.append({"id": key, "corpus": "prose", "turns": [split]})
    for key in sorted(eval_pool, key=_sha):
        text = eval_pool[key]
        title = key.split(":", 1)[1].rsplit("/", 1)[0] or key
        turns = chat_turns(title, text)
        if len(turns) >= 4 and int(_sha("chat:" + key)[:8], 16) % 2 == 0:
            if len(chat) < args.n_chat:
                chat.append({"id": key, "corpus": "chat", "turns": turns})
            continue
        split = _cut_third(text)
        if split and len(prose) < args.n_prose:
            prose.append({"id": key, "corpus": "prose", "turns": [split]})
    corpora["prose"] = prose
    corpora["chat"] = chat
    if args.quick:
        corpora = {k: v[:args.quick] for k, v in corpora.items()}
    return corpora, train_md


# ---------------------------------------------------------------------------------------
# proxy tokenizer
# ---------------------------------------------------------------------------------------

# The split regex family K3's tokenizer uses: case-aware letter runs with an optional
# leading non-letter, digit groups of at most three, punctuation runs carrying trailing
# newlines, newline runs, and whitespace that leaves one space for the next word.
SPLIT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _line_keys(text, min_len=24):
    return {hash(s) for s in (ln.strip() for ln in text.splitlines()) if len(s) >= min_len}


def tokenizer_training_files(args, eval_texts, train_md):
    """Training text for the proxy BPE, disjoint from every evaluation document.

    Sources that never feed evaluation: installed third-party Python packages, C
    headers, Perl modules and documentation, Go sources, and the tokenizer half of the
    package markdown. Any file sharing more than a quarter of its non-trivial lines with
    the evaluation text is dropped, which removes vendored or backported copies.
    """
    eval_lines = set()
    for t in eval_texts:
        eval_lines |= _line_keys(t)

    def admit(text):
        keys = _line_keys(text)
        return bool(keys) and len(keys & eval_lines) <= 0.25 * len(keys)

    picked, stats = [], {}

    def take(label, paths, budget):
        got = files = dropped = 0
        for p in sorted(paths, key=_sha):
            if got >= budget:
                break
            text = _read(p, 200000)
            if not text:
                continue
            if not admit(text):
                dropped += 1
                continue
            picked.append(text)
            got += len(text)
            files += 1
        stats[label] = {"files": files, "chars": got, "dropped_for_overlap": dropped}

    def walk(root, exts, skip=()):
        out = []
        for dp, _, fns in os.walk(root):
            if any(s in dp for s in skip):
                continue
            out += [os.path.join(dp, f) for f in fns if f.endswith(exts)]
        return out

    take("python_third_party", walk(args.site_packages, (".py",), ("distutils",)),
         args.train_py_chars)
    take("c_headers", walk("/usr/include", (".h",)), args.train_c_chars)
    take("perl_docs", walk("/usr/share/perl", (".pod", ".pm")), args.train_prose_chars)
    take("go_sources", [p for r in args.go_roots for p in walk(os.path.join(r, "src"),
                                                               (".go",), ("testdata",))],
         args.train_go_chars)
    kept_md = [t for t in train_md if admit(t)]
    picked += kept_md
    stats["package_markdown"] = {"files": len(kept_md), "chars": sum(map(len, kept_md)),
                                 "dropped_for_overlap": len(train_md) - len(kept_md)}
    return picked, stats


def train_tokenizer(texts, vocab_size, path):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tok.save(path)
    return tok


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------

def encode_docs(tok, docs):
    """ids per document with a minimal chat framing; returns (seq, segments, chars)."""
    V = tok.get_vocab_size()
    USER, ASSISTANT, END = V, V + 1, V + 2
    texts = [t for d in docs for turn in d["turns"] for t in turn]
    enc = tok.encode_batch(texts, add_special_tokens=False)
    it = iter(enc)
    out = []
    for d in docs:
        seq, segs, chars = [], [], 0
        for _prompt, answer in d["turns"]:
            p_ids, a_ids = next(it).ids, next(it).ids
            seq += [USER] + p_ids + [END, ASSISTANT]
            s = len(seq)
            seq += a_ids + [END]
            segs.append((s, len(seq)))
            chars += len(answer)
        out.append((seq, segs, chars))
    return out


def policy_configs(A, q3, only=None):
    """(name, family, params, nd per draft point, accepted prefix) for every config."""
    cfgs = []
    for N in (1, 2, 4, 8):
        cfgs.append(("P0/N%d" % N, "P0", {"N": N}, np.minimum(A["l0"], N), A["a0"]))
        cfgs.append(("P1/N%d" % N, "P1", {"N": N}, np.minimum(A["l1"], N), A["a1"]))
    for mn in range(2, 7):
        fire = A["L2"] >= mn
        for N in (1, 2, 4, 8):
            nd = np.where(fire, N, 0)
            cfgs.append(("P2/min%d/N%d" % (mn, N), "P2", {"min_len": mn, "N": N}, nd,
                         A["a2"]))
        nd = np.where(fire, np.minimum(A["g2"], K3_SPEC_MAX), 0)
        cfgs.append(("P2/min%d/N8/no-overlap" % mn, "P2x", {"min_len": mn, "N": 8}, nd,
                     A["a2"]))
    if q3 is not None:
        for delta in (0.1, 0.22, 0.35, 0.55, 0.75):
            for N in (4, 8):
                nd = np.minimum(adaptive_lengths(A["f3"], q3, delta), N)
                cfgs.append(("P3/d%.2f/N%d" % (delta, N), "P3", {"delta": delta, "N": N},
                             nd, A["a2"]))
            nd = adaptive_lengths(A["f3"], q3, delta, cumulative=False)
            cfgs.append(("P3cond/d%.2f/N8" % delta, "P3cond", {"delta": delta, "N": 8}, nd,
                         A["a2"]))
        for tier in TIERS:
            for mode in ("free", "today"):
                nd = cost_aware_lengths(A["f3"], q3, tier, replay=(mode == "today"))
                cfgs.append(("P3cost/%s/%s/N8" % (tier, mode), "P3cost",
                             {"tier": tier, "rule": mode, "N": 8}, nd, A["a2"]))
    if only is not None:
        cfgs = [c for c in cfgs if c[0] in only]
    return cfgs


def concat_arrays(parts):
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def get_tokenizer(args, vocab, training):
    """The proxy BPE for `vocab`, cached in the work directory under a key that covers
    everything that shapes it. `training` memoises the training text across vocabs."""
    from tokenizers import Tokenizer
    key = _sha(json.dumps([vocab, SPLIT_REGEX, args.site_packages, args.go_roots,
                           args.train_py_chars, args.train_c_chars, args.train_prose_chars,
                           args.train_go_chars, args.rev, args.quick]))[:12]
    path = os.path.join(args.work, "proxy-bpe-%d-%s.json" % (vocab, key))
    if os.path.exists(path) and os.path.exists(path + ".stats.json"):
        with open(path + ".stats.json") as f:
            return Tokenizer.from_file(path), json.load(f)
    training["trained"].append(vocab)
    if "texts" not in training:
        training["texts"], training["sources"] = tokenizer_training_files(
            args, training["eval_texts"], training["train_md"])
    t1 = time.time()
    tok = train_tokenizer(training["texts"], vocab, path)
    stats = {"sources": training["sources"], "chars": sum(map(len, training["texts"])),
             "requested_vocab": vocab, "train_seconds": round(time.time() - t1, 1)}
    with open(path + ".stats.json", "w") as f:
        json.dump(stats, f)
    return tok, stats


def analyse_corpora(corpora, tok, t0):
    """Every policy's drafts at every draft point, per document, per corpus."""
    per_corpus = {}
    for name, docs in corpora.items():
        parts, info = [], []
        for d, (seq, segs, chars) in zip(docs, encode_docs(tok, docs)):
            A = analyze_sequence(seq, segs)
            if len(A["rem"]) < 32:
                continue
            parts.append(A)
            info.append({"id": d["id"], "calib": is_calibration(d["id"]),
                         "points": len(A["rem"]),
                         "prompt_ids": len(seq) - sum(e - s for s, e in segs),
                         "output_ids": sum(e - s for s, e in segs), "output_chars": chars})
        per_corpus[name] = (parts, info)
        print("%-13s analysed %d documents, %d draft points (%.0f s)" % (
            name, len(parts), sum(i["points"] for i in info), time.time() - t0), flush=True)
    return per_corpus


def evaluate(per_corpus, boot, t0, only=None, q3=None):
    """Fit P3's table on the calibration documents (each corpus weighted equally), then
    replay every configuration on both splits. With `only`, just those configurations,
    evaluation split only. With `q3`, that table instead of a fitted one."""
    qinfo = None
    if q3 is None:
        cal = []
        for parts, info in per_corpus.values():
            sel = [p for p, i in zip(parts, info) if i["calib"]]
            if sel:
                cal.append(concat_arrays(sel))
        tot = sum(len(c["rem"]) for c in cal)
        weights = np.concatenate([np.full(len(c["rem"]), tot / (len(cal) * len(c["rem"])))
                                  for c in cal])
        calA = concat_arrays(cal)
        q3, qinfo = fit_acceptance_table(calA["f3"], calA["a2"], calA["rem"], weights)

    ev = {"q3": q3, "qinfo": qinfo, "results": {}, "calib": {}, "sums": {}, "p3_diag": {}}
    for name, (parts, info) in per_corpus.items():
        for split in ("eval",) if only else ("eval", "calib"):
            sel = [(p, i) for p, i in zip(parts, info) if i["calib"] == (split == "calib")]
            if not sel:
                continue
            A = concat_arrays([p for p, _ in sel])
            doc = np.concatenate([np.full(len(p["rem"]), k) for k, (p, _) in enumerate(sel)])
            W = bootstrap_weights(len(sel), boot, seed=int(_sha(name)[:8], 16))
            res = {}
            if split == "eval" and not only:
                ev["p3_diag"][name] = {t: p3_calibration(A["f3"], A["a2"], q3, DELTA[t])
                                       for t in TIERS}
            for cname, fam, params, nd, acc in policy_configs(A, q3, only):
                X = simulate(nd, acc, A["rem"], doc, len(sel))
                if split == "eval":
                    ev["sums"][(name, cname)] = (X, W)
                    res[cname] = {"family": fam, "params": params, **summarize(X, W)}
                else:
                    res[cname] = {k: float(v) for k, v in metrics(X.sum(axis=0)).items()}
            ev["results" if split == "eval" else "calib"][name] = res
        print("%-13s simulated (%.0f s)" % (name, time.time() - t0), flush=True)
    return ev


def run(args):
    t0 = time.time()
    os.makedirs(args.work, exist_ok=True)
    corpora, train_md = build_corpora(args)
    for name, docs in corpora.items():
        print("%-13s %4d documents" % (name, len(docs)), flush=True)
    training = {"eval_texts": [t for docs in corpora.values() for d in docs
                               for turn in d["turns"] for t in turn],
                "train_md": train_md, "trained": []}
    tok, tok_stats = get_tokenizer(args, args.vocab, training)
    print("tokenizer: vocab %d, trained on %.1f MB" % (tok.get_vocab_size(),
          tok_stats["chars"] / 1e6), flush=True)

    per_corpus = analyse_corpora(corpora, tok, t0)
    ev = evaluate(per_corpus, args.boot, t0)

    # P2 defaults chosen on the calibration split: the best mean speedup over corpora
    p2_choice = {}
    for tier in TIERS:
        key = "speedup_%s_free" % tier
        cal = ev["calib"]
        best = max((c for c in next(iter(cal.values())) if c.startswith("P2/")
                    and "no-overlap" not in c),
                   key=lambda c: np.mean([cal[k][c][key] for k in cal]))
        p2_choice[tier] = best

    out = assemble(args, per_corpus, tok, tok_stats, ev, p2_choice)
    out["p3_table_check_on_evaluation"] = ev["p3_diag"]

    # the same headline configurations under other proxy vocabularies
    names = {c for cfgs in headline_configs(p2_choice).values() for _, c in cfgs}
    sens = {}
    for vocab in args.sensitivity_vocab:
        tok_v, _stats_v = get_tokenizer(args, vocab, training)
        print("sensitivity tokenizer: vocab %d" % tok_v.get_vocab_size(), flush=True)
        pc_v = analyse_corpora(corpora, tok_v, t0)
        ev_v = evaluate(pc_v, args.boot, t0, only=names)
        keep = ("fire_rate", "drafted_per_fire", "accepted_per_fire", "acceptance_rate",
                "tokens_per_step", "speedup_8GB_free", "speedup_8GB_today",
                "speedup_128GB_free", "speedup_128GB_today")
        sens["vocab_%d" % tok_v.get_vocab_size()] = {
            "requested_vocab": vocab,
            "output_chars_per_id": {
                n: round(sum(i["output_chars"] for i in info if not i["calib"])
                         / max(1, sum(i["output_ids"] for i in info if not i["calib"])), 3)
                for n, (_p, info) in pc_v.items()},
            "results": {n: {c: {k: r[k] for k in keep} for c, r in res.items()}
                        for n, res in ev_v["results"].items()},
        }
    if sens:
        out["tokenizer_sensitivity"] = sens

    ru = resource.getrusage(resource.RUSAGE_SELF)
    out["resources"] = {"wall_seconds": round(time.time() - t0, 1),
                        "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 1),
                        "peak_rss_mb": round(ru.ru_maxrss / 1024.0, 1),
                        "tokenizers_trained_in_this_run": training["trained"]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(render_headline(out))
    print("wrote %s in %.0f s" % (args.out, time.time() - t0))


def headline_configs(p2_choice):
    """Per tier: the current rule and the eager rule at --spec 8, P2 with the defaults
    the calibration split chose for that tier, P3 at that tier's delta."""
    return {tier: [("P0", "P0/N8"), ("P1", "P1/N8"), ("P2", p2_choice[tier]),
                   ("P3", "P3/d%.2f/N8" % DELTA[tier]),
                   ("P3 cost-aware, today", "P3cost/%s/today/N8" % tier)] for tier in TIERS}


def assemble(args, per_corpus, tok, tok_stats, ev, p2_choice):
    q3, qinfo, results, sums = ev["q3"], ev["qinfo"], ev["results"], ev["sums"]
    corp_info = {}
    for name, (_parts, info) in per_corpus.items():
        evd = [i for i in info if not i["calib"]]
        corp_info[name] = {
            "documents_eval": len(evd),
            "documents_calibration": len(info) - len(evd),
            "prompt_ids_eval": sum(i["prompt_ids"] for i in evd),
            "output_ids_eval": sum(i["output_ids"] for i in evd),
            "draft_points_eval": sum(i["points"] for i in evd),
            "output_chars_per_id": round(sum(i["output_chars"] for i in evd)
                                         / max(1, sum(i["output_ids"] for i in evd)), 3),
        }
    corp_info["code_edits"]["source"] = (
        "this repository's non-merge history up to %s: each modified .c/.h/.py/.md file "
        "of 0.5-64 KB; prompt = old version + one instruction line naming the path and "
        "the commit subject, output = new version" % args.rev)
    corp_info["code_writing"]["source"] = (
        "Python standard library modules of 3-40 KB under %s, outside test directories; "
        "prompt = first third (to a line boundary), output = the rest" % args.stdlib)
    corp_info["prose"]["source"] = (
        "this repository's markdown of 2-60 KB and the evaluation half of the installed "
        "packages' markdown (npm, Ruby gems, Go); prompt = first third, output = the rest")
    corp_info["chat"]["source"] = (
        "technical Q&A synthesized from sectioned markdown in the evaluation half of the "
        "installed packages: one user question per heading, the section as the "
        "assistant's answer, up to 10 turns; every answer is output, every question and "
        "all earlier turns are history. No real conversation transcripts were used.")

    # paired bootstrap differences against the current rule, per tier and replay mode;
    # and the headline rows again with a per-verify overhead eps = 0.02
    diffs, eps_rows = {}, {}
    for tier, cfgs in headline_configs(p2_choice).items():
        for name in results:
            base_X, W = sums[(name, "P0/N8")]
            for label, cname in cfgs:
                X, _ = sums[(name, cname)]
                e = metrics(X.sum(0), eps=0.02)
                eps_rows.setdefault(tier, {}).setdefault(name, {})[label] = {
                    mode: round(float(e["speedup_%s_%s" % (tier, mode)]), 4)
                    for mode in ("free", "today")}
                if label == "P0":
                    continue
                row = {}
                for mode in ("free", "today"):
                    key = "speedup_%s_%s" % (tier, mode)
                    p = metrics(X.sum(0))[key] - metrics(base_X.sum(0))[key]
                    b = metrics(W @ X)[key] - metrics(W @ base_X)[key]
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    row[mode] = [round(float(p), 4), round(float(lo), 4), round(float(hi), 4)]
                diffs.setdefault(tier, {}).setdefault(name, {})[label + " - P0"] = row

    table = {"code_index": "(depth_bucket*%d + length_bucket)*%d + agreement_bucket"
             % (N_ELL, N_AGREE),
             "depth_buckets": ["1", "2-3", "4-8"],
             "length_bucket_lower_edges": list(_ELL_EDGES),
             "agreement_buckets": list(AGREE_NAMES),
             "global_rate": round(qinfo["root"], 4),
             "weighted_samples": round(qinfo["samples"], 1),
             "q": [round(float(x), 4) for x in q3]}

    return {
        "schema": 1,
        "kind": ("teacher-forced PROXY: human-written text stands in for K3's greedy "
                 "output and a byte-level BPE trained here stands in for K3's tokenizer; "
                 "a draft is accepted when it equals the next ids of the text. Not a "
                 "measurement of K3 and not a tokens-per-second figure."),
        "generated_by": "tools/spec_replay.py run",
        "git_rev": args.rev,
        "tokenizer": {
            "kind": "proxy byte-level BPE (tokenizers %s); K3's own tokenizer ships with "
                    "the checkpoint and was not available" % _tokenizers_version(),
            "vocab_size": tok.get_vocab_size(),
            "split_regex": SPLIT_REGEX,
            "training_chars": tok_stats["chars"],
            "training_sources": tok_stats["sources"],
            "disjointness": "no evaluation document or any copy of it is in the training "
                            "text: evaluation sources are this repository, the Python "
                            "standard library and the evaluation half of package "
                            "markdown (split by package); training files sharing more "
                            "than 25% of their lines of 24+ characters with any "
                            "evaluation text were dropped",
            "framing": "every turn is [user] prompt [end] [assistant] answer [end], the "
                       "three markers being ids outside the BPE vocabulary; the answer's "
                       "[end] is part of the output the drafts are checked against",
        },
        "cost_model": {
            "verify_step": "S(n) = a_T + a_E*u(n) + a_C*n + eps, n = drafted + 1",
            "u": "u(n) = 56*(1 - (55/56)^n)",
            "tiers": TIERS,
            "eps": 0.0,
            "no_draft_step": 1.0,
            "today": "a partial acceptance (m < nd) adds a_T + a_E*u(m+1) + a_C*(m+1)",
            "replay_free": "no second pass",
            "emitted_per_step": "m + 1, truncated at the end of the turn",
            "speedup": "emitted ids / summed step cost, pooled over the documents",
            "eps_sensitivity": "with a per-verify overhead eps, cost rises by "
                               "eps * fired steps: speedup_eps = emitted / (cost + "
                               "eps * fired), computable from these rows",
            "marginal_cost_first_draft": {t: round(marginal_cost(1, t), 4) for t in TIERS},
        },
        "method": {
            "draft_points": "every step the engine would take when its greedy output "
                            "equals the text: from each point it jumps by what that step "
                            "emits. The first id of each answer comes from prefill, where "
                            "the engine never drafts, and is excluded.",
            "split": "20% of each corpus's documents (hash of the document id) are "
                     "calibration: they fit P3's table and choose P2's defaults; every "
                     "reported row is on the other 80%",
            "bootstrap": "%d resamples of evaluation documents with replacement; 95%% "
                         "percentile intervals; the same resamples for every policy, so "
                         "differences are paired" % args.boot,
            "per_position_acceptance": "pos_i = P(draft i accepted | drafted and 1..i-1 "
                                       "accepted)",
        },
        "policies": {
            "P0": "spec_draft() as in src/cli/k3_run.c (port verified against crafted and "
                  "random sequences); N = --spec cap",
            "P1": "most recent occurrence of the suffix 4-gram, else 3-gram, else "
                  "2-gram; copy its continuation up to the end of the history; no gate",
            "P2": "longest suffix match (suffix automaton), most recent occurrence, fires "
                  "when match length >= min_len, drafts N ids, periodic copy past the "
                  "end of the history; the no-overlap variant stops at the history end",
            "P3": "P2's draft, length = leading j whose estimated P(1..j all accepted) > "
                  "delta, capped at N; P3cond uses P(j accepted | 1..j-1) > delta "
                  "instead of the product",
            "P3cost": "P3 with the threshold read off the cost model: add draft j while "
                      "P_j > S(j+1) - S(j), plus, for the 'today' rule, the expected "
                      "replay a rejection at j would add, (P_{j-1} - P_j) * R(j-1); "
                      "P3cost/<tier>/today is the variant that accounts for today's "
                      "replay",
        },
        "p2_defaults_from_calibration": p2_choice,
        "p3_table": table,
        "corpora": corp_info,
        "headline_configs": {t: dict(c) for t, c in headline_configs(p2_choice).items()},
        "results": results,
        "paired_speedup_differences_vs_P0": diffs,
        "headline_speedups_with_eps_0.02": eps_rows,
    }


def _tokenizers_version():
    try:
        import tokenizers
        return tokenizers.__version__
    except ImportError:
        return "?"


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    if pct:
        return "%.1f%%" % (100 * v[0])
    return "%.2f" % v[0]


def _ci(v):
    return "n/a" if v is None else "%.2f [%.2f, %.2f]" % tuple(v)


def render_headline(out):
    tables = []
    for tier, cfgs in out["headline_configs"].items():
        rows = ["%s tier" % tier, "",
                "| corpus | policy | config | fires | drafted/fire | accepted/fire | "
                "accepted/drafted | tokens/step | replay-free | today |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for corpus, res in out["results"].items():
            for label, cname in cfgs.items():
                r = res[cname]
                rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    corpus, label, cname, _fmt(r["fire_rate"], True),
                    _fmt(r["drafted_per_fire"]), _fmt(r["accepted_per_fire"]),
                    _fmt(r["acceptance_rate"], True), _fmt(r["tokens_per_step"]),
                    _ci(r["speedup_%s_free" % tier]), _ci(r["speedup_%s_today" % tier])))
        tables.append("\n".join(rows))
    return "\n\n".join(tables)


def load_id_docs(path):
    """Documents of token ids, one JSON object per line:
        {"id": "...", "turns": [[prompt_ids, output_ids], ...]}
    output_ids is what the engine emitted for the turn, ending with the id it stopped
    on; prompt_ids is everything fed before it (template and user text included)."""
    docs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            seq, segs = [], []
            for p_ids, o_ids in d["turns"]:
                seq += [int(x) for x in p_ids]
                s = len(seq)
                seq += [int(x) for x in o_ids]
                segs.append((s, len(seq)))
            docs.append((str(d.get("id", len(docs))), seq, segs))
    return docs


def replay_ids(args):
    """Replay the headline policies on ids the engine actually emitted (K3's tokenizer,
    K3's own greedy output): the measurement this proxy stands in for. P3 uses the table
    and P2 the defaults recorded in the reference JSON; every document is evaluated."""
    t0 = time.time()
    with open(args.reference) as f:
        ref = json.load(f)
    q3 = np.asarray(ref["p3_table"]["q"], float)
    parts, info = [], []
    for doc_id, seq, segs in load_id_docs(args.file):
        A = analyze_sequence(seq, segs)
        if len(A["rem"]):
            parts.append(A)
            info.append({"id": doc_id, "calib": False, "points": len(A["rem"])})
    names = {c for cfg in ref["headline_configs"].values() for c in cfg.values()}
    ev = evaluate({"ids": (parts, info)}, args.boot, t0, only=names, q3=q3)
    out = {"kind": "replay of emitted ids", "reference": args.reference,
           "documents": len(parts), "draft_points": sum(i["points"] for i in info),
           "headline_configs": ref["headline_configs"], "results": ev["results"]}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    print(render_headline(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build corpora, train the proxy tokenizer, replay")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r.add_argument("--work", required=True, help="scratch directory (tokenizer cache)")
    r.add_argument("--out", default=os.path.join(here, "docs", "measurements",
                                                 "spec-replay.json"))
    r.add_argument("--repo", default=here)
    r.add_argument("--rev", default="HEAD")
    r.add_argument("--stdlib", default="/usr/lib/python3.11")
    r.add_argument("--site-packages", default="/usr/local/lib/python3.11/dist-packages")
    r.add_argument("--node-roots", nargs="*", default=[
        "/opt/node22/lib/node_modules", "/opt/node21/lib/node_modules",
        "/opt/node20/lib/node_modules"])
    r.add_argument("--ruby-roots", nargs="*", default=[
        "/opt/ruby-3.3.6/lib/ruby/gems", "/opt/ruby-3.2.6/lib/ruby/gems",
        "/opt/ruby-3.1.6/lib/ruby/gems"])
    r.add_argument("--go-roots", nargs="*", default=["/usr/local/go1.25.1"])
    r.add_argument("--n-writing", type=int, default=220)
    r.add_argument("--n-prose", type=int, default=220)
    r.add_argument("--n-chat", type=int, default=140)
    r.add_argument("--vocab", type=int, default=131072)
    r.add_argument("--train-py-chars", type=int, default=24_000_000)
    r.add_argument("--train-c-chars", type=int, default=12_000_000)
    r.add_argument("--train-prose-chars", type=int, default=10_000_000)
    r.add_argument("--train-go-chars", type=int, default=10_000_000)
    r.add_argument("--sensitivity-vocab", type=int, nargs="*", default=[32768],
                   help="also replay the headline policies under these proxy vocabularies")
    r.add_argument("--boot", type=int, default=2000)
    r.add_argument("--quick", type=int, default=0, help="documents per corpus (debug)")
    i = sub.add_parser("replay-ids", help="replay the headline policies on emitted ids")
    i.add_argument("file", help="JSON lines: {\"id\", \"turns\": [[prompt_ids, output_ids]]}")
    i.add_argument("--reference", default=os.path.join(here, "docs", "measurements",
                                                       "spec-replay.json"))
    i.add_argument("--out")
    i.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.cmd == "replay-ids":
        replay_ids(args)
        return 0
    if args.cmd == "run":
        args.rev = _git(args.repo, "rev-parse", args.rev).decode().strip()
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
