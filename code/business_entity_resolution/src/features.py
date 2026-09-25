#!/usr/bin/env python3
"""
Feature Engineering Module for Business Entity Resolution.

Implements rich evidence vectors for (Source 1, Candidate) pairs:
1. Multi-metric Name Similarities (Jaccard, trigrams, Levenshtein, token-sort, token-set, partial, WRatio, exact)
2. Multi-metric Address Similarities (Jaccard, trigrams, Levenshtein, token-sort, token-set, partial)
3. Structural & Postal/Numeric Features (postal match/mismatch/availability, numeric token overlap)
4. Evidence Availability & Missingness Indicators (both present, one missing, neither)
5. Cross-field Interactions & Contradiction Penalties (name*addr, name*(1-addr), postal mismatch flag)
6. Phonetic Matching
"""
import re, math
import numpy as np
from rapidfuzz import fuzz
from metaphone import doublemetaphone

# Feature list definition
FEATURE_NAMES = [
    # ── 1. Name Similarities (12) ──
    'name_jaccard',
    'name_tri_jacc',
    'name_lev',
    'name_tsort',
    'name_tset',
    'name_partial',
    'name_wr',
    'name_exact',
    'name_exact_sorted',
    'name_len_d',
    'name_len_r',
    'name_tok_d',
    
    # ── 2. Address Similarities (11) ──
    'addr_jaccard',
    'addr_tri_jacc',
    'addr_lev',
    'addr_tsort',
    'addr_tset',
    'addr_partial',
    'addr_len_d',
    'addr_len_r',
    'addr_tok_d',
    'addr_exact',
    'addr_exact_sorted',
    
    # ── 3. Postal & Numeric Components (5) ──
    'post_match',
    'post_mismatch',
    'post_avail',
    'num_jacc',
    'num_match_count',
    
    # ── 4. Evidence Availability & Missingness (6) ──
    'both_name_present',
    'both_addr_present',
    's1_addr_missing',
    'cand_addr_missing',
    'only_one_addr_present',
    'num_available_fields',
    
    # ── 5. Cross-field Interactions & Contradictions (10) ──
    'country_match',
    'comb_tsort',
    'comb_tset',
    'name_addr_prod_tsort',
    'name_addr_prod_tset',
    'name_addr_prod_jacc',
    'name_addr_gap',
    'addr_contradiction',
    'name_high_addr_zero',
    'strong_evidence_both',
    
    # ── 6. Phonetic & Prefix Features (4) ──
    'phone_first_m',
    'phone_all_jacc',
    'pfx3_match',
    'pfx5_match',
]

def _tri(s: str) -> set:
    if not s or len(s) < 3: return set()
    return set(s[i:i+3] for i in range(len(s) - 2))

def get_phone_set(norm_name_str: str) -> set:
    if not norm_name_str: return set()
    res = set()
    for tok in norm_name_str.split():
        if len(tok) >= 2:
            try:
                p, s = doublemetaphone(tok)
                if p: res.add(p)
                if s: res.add(s)
            except Exception:
                pass
    return res

def compute_pair_features(s1_nn: str, s1_na: str, s1_c: str, s1_rn: str, s1_ra: str,
                          c_nn: str, c_na: str, c_c: str, c_rn: str, c_ra: str,
                          s1_pc: str = "", c_pc: str = "") -> np.ndarray:
    """
    Computes all 48 pairwise similarity, structural, missingness, and interaction features.
    """
    # ── 1. Name Features ──
    s1t = set(s1_nn.split()) if s1_nn else set()
    ct  = set(c_nn.split()) if c_nn else set()
    name_jacc = len(s1t & ct) / max(len(s1t | ct), 1) if (s1t or ct) else 0.0
    
    sg, cg = _tri(s1_nn), _tri(c_nn)
    name_tri_jacc = len(sg & cg) / max(len(sg | cg), 1) if (sg or cg) else 0.0
    
    name_lev     = fuzz.ratio(s1_nn, c_nn) / 100.0 if (s1_nn or c_nn) else 0.0
    name_tsort   = fuzz.token_sort_ratio(s1_nn, c_nn) / 100.0 if (s1_nn or c_nn) else 0.0
    name_tset    = fuzz.token_set_ratio(s1_nn, c_nn) / 100.0 if (s1_nn or c_nn) else 0.0
    name_partial = fuzz.partial_ratio(s1_nn, c_nn) / 100.0 if (s1_nn or c_nn) else 0.0
    name_wr      = fuzz.WRatio(s1_nn, c_nn) / 100.0 if (s1_nn or c_nn) else 0.0
    
    name_exact = 1.0 if (s1_nn and s1_nn == c_nn) else 0.0
    s1s = ' '.join(sorted(s1_nn.split())) if s1_nn else ''
    cs  = ' '.join(sorted(c_nn.split())) if c_nn else ''
    name_exact_sorted = 1.0 if (s1s and s1s == cs) else 0.0
    
    name_len_d = abs(len(s1_nn) - len(c_nn))
    name_len_r = min(len(s1_nn), len(c_nn)) / max(len(s1_nn), len(c_nn), 1)
    name_tok_d = abs(len(s1t) - len(ct))
    
    # ── 2. Address Features ──
    s1at = set(s1_na.split()) if s1_na else set()
    cat  = set(c_na.split()) if c_na else set()
    addr_jacc = len(s1at & cat) / max(len(s1at | cat), 1) if (s1at or cat) else 0.0
    
    sag, cag = _tri(s1_na), _tri(c_na)
    addr_tri_jacc = len(sag & cag) / max(len(sag | cag), 1) if (sag or cag) else 0.0
    
    addr_lev     = fuzz.ratio(s1_na, c_na) / 100.0 if (s1_na or c_na) else 0.0
    addr_tsort   = fuzz.token_sort_ratio(s1_na, c_na) / 100.0 if (s1_na or c_na) else 0.0
    addr_tset    = fuzz.token_set_ratio(s1_na, c_na) / 100.0 if (s1_na or c_na) else 0.0
    addr_partial = fuzz.partial_ratio(s1_na, c_na) / 100.0 if (s1_na or c_na) else 0.0
    
    addr_len_d = abs(len(s1_na) - len(c_na))
    addr_len_r = min(len(s1_na), len(c_na)) / max(len(s1_na), len(c_na), 1)
    addr_tok_d = abs(len(s1at) - len(cat))
    addr_exact = 1.0 if (s1_na and s1_na == c_na) else 0.0
    s1as = ' '.join(sorted(s1_na.split())) if s1_na else ''
    cas  = ' '.join(sorted(c_na.split())) if c_na else ''
    addr_exact_sorted = 1.0 if (s1as and s1as == cas) else 0.0
    
    # ── 3. Postal & Numeric Components ──
    s1p = s1_pc if s1_pc else ("" if not s1_ra else "")
    cp_ = c_pc if c_pc else ("" if not c_ra else "")
    post_avail = 1.0 if (s1p and cp_) else 0.0
    post_match = 1.0 if (post_avail and s1p == cp_) else 0.0
    post_mismatch = 1.0 if (post_avail and s1p != cp_) else 0.0
    
    sn = set(re.findall(r'\d+', s1_ra)) if s1_ra else set()
    cn = set(re.findall(r'\d+', c_ra)) if c_ra else set()
    num_jacc = len(sn & cn) / max(len(sn | cn), 1) if (sn or cn) else 0.0
    num_match_count = float(len(sn & cn))
    
    # ── 4. Evidence Availability & Missingness ──
    both_name_present = 1.0 if (s1_nn and c_nn) else 0.0
    both_addr_present = 1.0 if (s1_na and c_na) else 0.0
    s1_addr_missing   = 1.0 if not s1_na else 0.0
    cand_addr_missing = 1.0 if not c_na else 0.0
    only_one_addr_present = 1.0 if ((s1_na and not c_na) or (not s1_na and c_na)) else 0.0
    
    avail_count = 0.0
    if s1_nn and c_nn: avail_count += 1.0
    if s1_na and c_na: avail_count += 1.0
    if s1p and cp_: avail_count += 1.0
    if s1_c and c_c: avail_count += 1.0
    num_available_fields = avail_count
    
    # ── 5. Cross-field Interactions & Contradictions ──
    country_match = 1.0 if (s1_c and c_c and s1_c == c_c) else 0.0
    comb_tsort = (name_tsort + addr_tsort) / 2.0
    comb_tset  = (name_tset + addr_tset) / 2.0
    
    name_addr_prod_tsort = name_tsort * addr_tsort
    name_addr_prod_tset  = name_tset * addr_tset
    name_addr_prod_jacc  = name_jacc * addr_jacc
    
    name_addr_gap = abs(name_tsort - addr_tsort)
    
    # Contradiction: both addresses exist, long (>8 chars), but almost 0 similarity
    if both_addr_present and len(s1_na) > 8 and len(c_na) > 8 and addr_tsort < 0.25 and num_jacc == 0.0 and post_mismatch:
        addr_contradiction = 1.0
    elif both_addr_present and len(s1_na) > 8 and len(c_na) > 8 and addr_tsort < 0.20 and num_jacc == 0.0:
        addr_contradiction = 0.8
    else:
        addr_contradiction = 0.0
        
    name_high_addr_zero = 1.0 if (name_tsort > 0.85 and cand_addr_missing) else 0.0
    strong_evidence_both = 1.0 if (name_tsort > 0.80 and addr_tsort > 0.75) else 0.0
    
    # ── 6. Phonetic & Prefix Features ──
    phone_first_m = 0.0
    try:
        f1 = s1_nn.split()[0] if s1_nn else ''
        f2 = c_nn.split()[0] if c_nn else ''
        if f1 and f2:
            p1, _ = doublemetaphone(f1)
            p2, _ = doublemetaphone(f2)
            if p1 and p2 and p1 == p2:
                phone_first_m = 1.0
    except Exception:
        pass
        
    p1_set = get_phone_set(s1_nn)
    p2_set = get_phone_set(c_nn)
    phone_all_jacc = len(p1_set & p2_set) / max(len(p1_set | p2_set), 1) if (p1_set or p2_set) else 0.0
    
    pfx3_match = 1.0 if (len(s1_nn) >= 3 and len(c_nn) >= 3 and s1_nn[:3] == c_nn[:3]) else 0.0
    pfx5_match = 1.0 if (len(s1_nn) >= 5 and len(c_nn) >= 5 and s1_nn[:5] == c_nn[:5]) else 0.0
    
    return np.array([
        name_jacc, name_tri_jacc, name_lev, name_tsort, name_tset, name_partial,
        name_wr, name_exact, name_exact_sorted, name_len_d, name_len_r, name_tok_d,
        addr_jacc, addr_tri_jacc, addr_lev, addr_tsort, addr_tset, addr_partial,
        addr_len_d, addr_len_r, addr_tok_d, addr_exact, addr_exact_sorted,
        post_match, post_mismatch, post_avail, num_jacc, num_match_count,
        both_name_present, both_addr_present, s1_addr_missing, cand_addr_missing,
        only_one_addr_present, num_available_fields,
        country_match, comb_tsort, comb_tset,
        name_addr_prod_tsort, name_addr_prod_tset, name_addr_prod_jacc,
        name_addr_gap, addr_contradiction, name_high_addr_zero, strong_evidence_both,
        phone_first_m, phone_all_jacc, pfx3_match, pfx5_match
    ], dtype=np.float32)
