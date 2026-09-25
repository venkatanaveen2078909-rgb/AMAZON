#!/usr/bin/env python3
r"""
Production Inference Pipeline for ML Challenge 2026: Business Entity Resolution.
Generates:
  1. output/matching_results.tsv
  2. output/candidate_pairs.tsv

Validated with official utils/validate_submission.py
"""
import os
import sys
import gc
import re
import time
import argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from metaphone import doublemetaphone

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import CACHE_DIR, OUTPUT_DIR
from features import FEATURE_NAMES, compute_pair_features

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STUDENT_RES_DIR = os.path.join(ROOT_DIR, "6ab10eb3b23ba_student_resource", "student_resource")
TEST_DIR = os.path.join(STUDENT_RES_DIR, "dataset", "test")

# Calibrated Decision Parameters from EXP-04
BASE_THR = 0.97
MISSING_ADDR_THR = 0.99
HIGH_CONF_THR = 0.85
MARGIN = 0.15
MAX_CANDS_PER_S1 = 50

# ── High-Speed Normalization Regexes ──────────────────────────────────────────
_LEGAL = re.compile(
    r'\b(llc|llp|inc|incorporated|corp|corporation|co|company|ltd|limited|plc|lp|'
    r'group|enterprise|enterprises|holding|holdings|pvt|private|ngo|trust|foundation|'
    r'society|opc|sarl|sa|sas|sasu|eurl|snc|sca|sci|gie|&|and|et)\b', re.IGNORECASE)

_ADDR_RE = re.compile(
    r'\b(street|road|avenue|boulevard|drive|lane|court|place|highway|suite|apartment|building|floor)\b',
    re.IGNORECASE)
_ADDR_MAP = {
    'street': 'st', 'road': 'rd', 'avenue': 'ave', 'boulevard': 'blvd',
    'drive': 'dr', 'lane': 'ln', 'court': 'ct', 'place': 'pl',
    'highway': 'hwy', 'suite': 'ste', 'apartment': 'apt', 'building': 'bldg', 'floor': 'fl'
}

_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_SPACES = re.compile(r'\s+')
_NUMS = re.compile(r'\d+')
_POSTAL = re.compile(r'\b(\d{5,6})\b')

ADDR_STOPWORDS = {
    'road', 'street', 'lane', 'avenue', 'floor', 'building', 'phase', 'block',
    'sector', 'near', 'opposite', 'behind', 'beside', 'dist', 'state', 'city',
    'delhi', 'mumbai', 'pune', 'bangalore', 'chennai', 'kolkata', 'hyderabad',
    'india', 'france', 'paris', 'lyon', 'bordeaux', 'texas', 'california',
    'florida', 'york', 'north', 'south', 'east', 'west', 'unit', 'suite',
    'nagar', 'colony', 'marg', 'bhavan', 'tower', 'complex', 'plaza', 'bldg',
    'cross', 'main', 'extn', 'hno', 'plot', 'shop', 'flat'
}

def fast_norm_name(s: str) -> str:
    if not s: return ""
    s = str(s).lower().strip()
    s = _LEGAL.sub(' ', s)
    s = _PUNCT.sub(' ', s)
    return _SPACES.sub(' ', s).strip()

def fast_norm_name_sorted(nn: str) -> str:
    if not nn: return ""
    tokens = nn.split()
    tokens.sort()
    return ' '.join(tokens)

def fast_norm_addr(s: str) -> str:
    if not s: return ""
    s = str(s).lower().strip()
    s = _ADDR_RE.sub(lambda m: _ADDR_MAP[m.group(0).lower()], s)
    s = _PUNCT.sub(' ', s)
    return _SPACES.sub(' ', s).strip()

def fast_get_postal(s: str) -> str:
    if not s: return ""
    m = _POSTAL.findall(str(s))
    return m[-1] if m else ""

def fast_get_nums(s: str) -> list:
    if not s: return []
    return _NUMS.findall(str(s))

# Phonetic Vocabulary Memoizer
class PhoneticMemoizer:
    def __init__(self):
        self.memo = {}

    def get_keys(self, text: str) -> tuple:
        if not text: return ()
        keys = []
        for tok in str(text).lower().split():
            if len(tok) >= 2:
                if tok not in self.memo:
                    try:
                        p, s = doublemetaphone(tok)
                        self.memo[tok] = (p, s) if s and s != p else ((p,) if p else ())
                    except Exception:
                        self.memo[tok] = ()
                for k in self.memo[tok]:
                    keys.append(k)
        return tuple(keys)

memoizer = PhoneticMemoizer()

def get_rare_tokens(norm_name_str: str) -> tuple:
    if not norm_name_str: return ()
    return tuple(w for w in norm_name_str.split() if len(w) >= 3)

def get_rare_addr_tokens(na_str: str) -> tuple:
    if not na_str: return ()
    return tuple(w for w in na_str.split() if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS)

def get_addr_keys(na_str: str) -> tuple:
    if not na_str: return ()
    nums = fast_get_nums(na_str)
    toks = [w for w in na_str.split() if len(w) >= 3 and not w.isdigit()]
    if nums and toks:
        if len(toks) > 1:
            return (f"{nums[0]}_{toks[0]}", f"{nums[0]}_{toks[-1]}")
        return (f"{nums[0]}_{toks[0]}",)
    return ()

def get_name_addr_composite(nn_str: str, na_str: str) -> tuple:
    if not nn_str or not na_str: return ()
    pfx2 = nn_str[:2]
    nums = fast_get_nums(na_str)
    if pfx2 and nums:
        return (f"{pfx2}_{nums[0]}",)
    return ()


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit S1 entities for testing")
    parser.add_argument("--chunk-size", type=int, default=30000, help="Chunk size for streaming S1")
    args = parser.parse_args()

    print("=" * 75, flush=True)
    print("PRODUCTION INFERENCE PIPELINE — ML CHALLENGE 2026", flush=True)
    print("=" * 75, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matching_out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    # 1. Load Model
    model_path = os.path.join(CACHE_DIR, "exp03_model.txt")
    print(f"Loading trained LightGBM model from {model_path}...", flush=True)
    model = lgb.Booster(model_file=model_path)

    # 2. Load Candidate Datasets (Test S2 + S3)
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")

    print(f"\n[1/4] Loading candidate datasets...", flush=True)
    t0 = time.time()
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"  Loaded {len(s2):,} S2 records ({time.time()-t0:.1f}s).", flush=True)
    
    t0 = time.time()
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"  Loaded {len(s3):,} S3 records ({time.time()-t0:.1f}s).", flush=True)

    cand_df = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    n_cands = len(cand_df)
    print(f"  Total candidate pool: {n_cands:,} records.", flush=True)

    # 3. High-Speed Normalization & Inverted Indexing
    print(f"\n[2/4] High-speed normalization & token frequency profiling...", flush=True)
    t_norm = time.time()
    c_ids = cand_df["entity_id"].values
    c_raw_names = cand_df["business_name"].values
    c_raw_addrs = cand_df["business_address"].values
    c_countries = cand_df["country"].values

    c_nn = [fast_norm_name(x) for x in c_raw_names]
    c_nns = [fast_norm_name_sorted(nn) for nn in c_nn]
    c_na = [fast_norm_addr(x) for x in c_raw_addrs]
    c_pc = [fast_get_postal(x) for x in c_raw_addrs]

    print(f"  Normalized {n_cands:,} candidate records in {time.time() - t_norm:.1f}s.", flush=True)

    print(f"\n[3/4] Building 8 inverted blocking indices...", flush=True)
    t_idx = time.time()
    idx_exact = defaultdict(list)
    idx_pfx3 = defaultdict(list)
    idx_phone = defaultdict(list)
    idx_postal = defaultdict(list)
    idx_addr_key = defaultdict(list)
    idx_comp = defaultdict(list)

    name_tok_cnt = Counter()
    addr_tok_cnt = Counter()

    for i in range(n_cands):
        # Exact
        nns_v = c_nns[i]
        if nns_v: idx_exact[nns_v].append(i)
        
        # Pfx3
        nn_v = c_nn[i]
        if len(nn_v) >= 3: idx_pfx3[nn_v[:3]].append(i)

        # Phone (memoized)
        for pk in memoizer.get_keys(c_raw_names[i]):
            idx_phone[pk].append(i)

        # Postal
        pc_v = c_pc[i]
        if pc_v: idx_postal[pc_v].append(i)

        # Addr keys
        na_v = c_na[i]
        for ak in get_addr_keys(na_v):
            idx_addr_key[ak].append(i)

        # Composite
        for ck in get_name_addr_composite(nn_v, na_v):
            idx_comp[ck].append(i)

        # Token frequencies
        for w in get_rare_tokens(nn_v):
            name_tok_cnt[w] += 1
        for w in get_rare_addr_tokens(na_v):
            addr_tok_cnt[w] += 1

    # Build rare token indices
    idx_rare_tok = defaultdict(list)
    idx_rare_addr = defaultdict(list)
    for i in range(n_cands):
        nn_v = c_nn[i]
        for w in get_rare_tokens(nn_v):
            if name_tok_cnt[w] <= 10000:
                idx_rare_tok[w].append(i)
        na_v = c_na[i]
        for w in get_rare_addr_tokens(na_v):
            if addr_tok_cnt[w] <= 600:
                idx_rare_addr[w].append(i)

    # Freeze dicts
    idx_exact = dict(idx_exact)
    idx_pfx3 = dict(idx_pfx3)
    idx_phone = dict(idx_phone)
    idx_postal = dict(idx_postal)
    idx_addr_key = dict(idx_addr_key)
    idx_comp = dict(idx_comp)
    idx_rare_tok = dict(idx_rare_tok)
    idx_rare_addr = dict(idx_rare_addr)

    del name_tok_cnt, addr_tok_cnt, cand_df
    gc.collect()
    print(f"  8 inverted indices built in {time.time() - t_idx:.1f}s. Memory optimized.", flush=True)

    # 4. Stream Source 1 in Batches
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    print(f"\n[4/4] Processing {s1_path} in streaming batches of {args.chunk_size:,}...", flush=True)

    s1_reader = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False,
                            encoding="utf-8", chunksize=args.chunk_size)

    total_s1 = 0
    total_cand_pairs = 0
    total_matches = 0

    with open(matching_out_path, "w", encoding="utf-8", newline="\n") as f_match, \
         open(candidate_out_path, "w", encoding="utf-8", newline="\n") as f_cand:

        # Exact headers required by official submission validator
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        batch_idx = 0
        for s1_chunk in s1_reader:
            batch_idx += 1
            b_start = time.time()
            n_chunk = len(s1_chunk)

            s1_ids = s1_chunk["entity_id"].values
            s1_raw_names = s1_chunk["business_name"].values
            s1_raw_addrs = s1_chunk["business_address"].values
            s1_countries = s1_chunk["country"].values

            s1_nn = [fast_norm_name(x) for x in s1_raw_names]
            s1_nns = [fast_norm_name_sorted(nn) for nn in s1_nn]
            s1_na = [fast_norm_addr(x) for x in s1_raw_addrs]
            s1_pc = [fast_get_postal(x) for x in s1_raw_addrs]

            # Step A: Blocking Query
            batch_pairs = [] # (s1_local_idx, cand_int_idx)
            s1_cand_sets = []

            for i in range(n_chunk):
                s_cands = set(idx_exact.get(s1_nns[i], []))

                nn_v = s1_nn[i]
                pfx3 = nn_v[:3] if len(nn_v) >= 3 else nn_v
                if pfx3:
                    b = idx_pfx3.get(pfx3, [])
                    if len(b) <= 800: s_cands.update(b)

                for pk in memoizer.get_keys(s1_raw_names[i]):
                    b = idx_phone.get(pk, [])
                    if len(b) <= 800: s_cands.update(b)

                for w in get_rare_tokens(nn_v):
                    b = idx_rare_tok.get(w, [])
                    if len(b) <= 800: s_cands.update(b)

                pc_v = s1_pc[i]
                if pc_v:
                    s_cands.update(idx_postal.get(pc_v, []))

                na_v = s1_na[i]
                for ak in get_addr_keys(na_v):
                    b = idx_addr_key.get(ak, [])
                    if len(b) <= 400: s_cands.update(b)

                for ck in get_name_addr_composite(nn_v, na_v):
                    b = idx_comp.get(ck, [])
                    if len(b) <= 400: s_cands.update(b)

                for w in get_rare_addr_tokens(na_v):
                    b = idx_rare_addr.get(w, [])
                    if len(b) <= 400: s_cands.update(b)

                # Cap candidates per S1
                if len(s_cands) > MAX_CANDS_PER_S1:
                    c_list = list(s_cands)[:MAX_CANDS_PER_S1]
                else:
                    c_list = list(s_cands)

                s1_cand_sets.append({c_ids[j] for j in c_list})
                for j in c_list:
                    batch_pairs.append((i, j))

            n_pairs = len(batch_pairs)
            total_cand_pairs += n_pairs
            s1_matches = [[] for _ in range(n_chunk)]

            # Step B: Feature Computation & Model Prediction
            if n_pairs > 0:
                X_batch = np.zeros((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)
                pair_meta = []

                for p_idx, (s1_i, cand_j) in enumerate(batch_pairs):
                    feat = compute_pair_features(
                        s1_nn[s1_i], s1_na[s1_i], s1_countries[s1_i],
                        s1_raw_names[s1_i], s1_raw_addrs[s1_i],
                        c_nn[cand_j], c_na[cand_j], c_countries[cand_j],
                        c_raw_names[cand_j], c_raw_addrs[cand_j],
                        s1_pc[s1_i], c_pc[cand_j]
                    )
                    X_batch[p_idx] = feat
                    # (s1_i, cand_id, both_addr, cand_miss, n_tsort, a_tset, contra)
                    pair_meta.append((s1_i, c_ids[cand_j], feat[29], feat[31], feat[3], feat[15], feat[41]))

                probs = model.predict(X_batch)
                del X_batch

                # Step C: Tiered Evidence Calibration
                s1_scores = defaultdict(list)
                for p_idx, p_score in enumerate(probs):
                    s1_i, cid, both_a, cand_miss, n_tsort, a_tset, contra = pair_meta[p_idx]

                    # 1. Contradiction Veto Filter
                    if contra >= 0.8 and p_score < 0.995:
                        continue

                    # 2. Tiered Thresholds
                    if both_a == 1.0 and n_tsort >= 0.80 and a_tset >= 0.80:
                        eff_thr = HIGH_CONF_THR
                    elif cand_miss == 1.0:
                        eff_thr = MISSING_ADDR_THR
                    else:
                        eff_thr = BASE_THR

                    if p_score >= eff_thr:
                        s1_scores[s1_i].append((cid, p_score))

                # Step D: Relative Margin Pruning
                for s1_i, scored_cands in s1_scores.items():
                    if not scored_cands:
                        continue
                    max_p = max(p for _, p in scored_cands)
                    accepted = [cid for cid, p in scored_cands if p >= (max_p - MARGIN)]
                    s1_matches[s1_i] = accepted
                    total_matches += len(accepted)

            # Step E: Output Writing (with matched ⊆ candidates guarantee)
            for i in range(n_chunk):
                sid = s1_ids[i]
                c_set = s1_cand_sets[i]
                m_list = s1_matches[i]

                # Strict subset guarantee
                for mid in m_list:
                    c_set.add(mid)

                m_str = ",".join(m_list)
                c_str = ",".join(sorted(c_set))

                f_match.write(f"{sid}\t{m_str}\n")
                f_cand.write(f"{sid}\t{c_str}\n")

            total_s1 += n_chunk
            print(f"  Batch {batch_idx:3d}: {total_s1:,} / 1,732,544 S1 entities "
                  f"({n_pairs:,} pairs scored, {len(m_list)} sample matches, "
                  f"{time.time() - b_start:.1f}s)", flush=True)

            if args.limit and total_s1 >= args.limit:
                print(f"  Reached limit of {args.limit:,} S1 entities.", flush=True)
                break

    print("=" * 75, flush=True)
    print(f"INFERENCE FINISHED IN {(time.time() - t_start)/60:.2f} MINUTES", flush=True)
    print(f"  Total S1 Entities:     {total_s1:,}", flush=True)
    print(f"  Total Candidate Pairs: {total_cand_pairs:,}", flush=True)
    print(f"  Total Matches:         {total_matches:,}", flush=True)
    print(f"  Matching Output:       {matching_out_path}", flush=True)
    print(f"  Candidate Output:      {candidate_out_path}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
