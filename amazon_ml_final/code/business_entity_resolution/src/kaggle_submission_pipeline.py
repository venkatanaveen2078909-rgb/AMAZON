#!/usr/bin/env python3
r"""
================================================================================
Amazon ML Challenge 2026: Business Entity Resolution
Production End-to-End Pipeline & Kaggle Execution Script (Memory-Optimized)
================================================================================
Configuration: EXP-04 (8-Channel Blocking + 48 Features GBDT + Tiered Decision Rules)
Validation Score: Macro F0.5 = 0.9456 (Precision = 0.9668, Singleton Acc = 0.9077)

Works seamlessly across:
  - Kaggle Notebooks (/kaggle/input/ and /kaggle/working/)
  - Local workstation
  - Linux/Cloud Servers
================================================================================
"""
import os
import sys
import gc
import re
import math
import time
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz
from metaphone import doublemetaphone

try:
    import psutil
except ImportError:
    psutil = None

def get_ram_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0

# UTF-8 Configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Environment & Path Resolution ─────────────────────────────────────────────
def resolve_paths():
    """Detect runtime environment and set standard paths."""
    kaggle_input = Path("/kaggle/input")
    kaggle_working = Path("/kaggle/working")
    
    if kaggle_input.exists():
        print("[Env] Running in Kaggle environment.", flush=True)
        ds_matches = list(kaggle_input.glob("**/test_source1.tsv"))
        if ds_matches:
            test_dir = ds_matches[0].parent
            train_dir = test_dir.parent / "train" if (test_dir.parent / "train").exists() else test_dir
        else:
            test_dir = kaggle_input / "amazon-ml-challenge-2026" / "dataset" / "test"
            train_dir = kaggle_input / "amazon-ml-challenge-2026" / "dataset" / "train"
            
        output_dir = kaggle_working / "output"
        cache_dir = kaggle_working / "cache"
        model_dir = kaggle_working / "models"
    else:
        print("[Env] Running in Local / Server environment.", flush=True)
        curr = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        root = curr
        for _ in range(4):
            if (root / "6ab10eb3b23ba_student_resource").exists():
                break
            root = root.parent
            
        base_res = root / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
        if base_res.exists():
            test_dir = base_res / "test"
            train_dir = base_res / "train"
        else:
            test_dir = root / "dataset" / "test"
            train_dir = root / "dataset" / "train"
            
        output_dir = root / "output"
        cache_dir = root / "code" / "business_entity_resolution" / "cache"
        model_dir = root / "models"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    return train_dir, test_dir, output_dir, cache_dir, model_dir

# ── Feature Definitions (48 Pairwise Dimensions) ──────────────────────────────
FEATURE_NAMES = [
    # 1. Name Similarities (12)
    'name_jaccard', 'name_tri_jacc', 'name_lev', 'name_tsort', 'name_tset',
    'name_partial', 'name_wr', 'name_exact', 'name_exact_sorted', 'name_len_d',
    'name_len_r', 'name_pfx3_eq',
    # 2. Address Similarities (10)
    'addr_jaccard', 'addr_tri_jacc', 'addr_lev', 'addr_tsort', 'addr_tset',
    'addr_partial', 'addr_num_exact', 'num_jacc', 'addr_len_d', 'addr_len_r',
    # 3. Phonetic Similarities (4)
    'phon_overlap', 'phon_jacc', 'addr_phon_overlap', 'addr_phon_jacc',
    # 4. Postal Structural Features (3)
    'postal_match', 'postal_mismatch', 'postal_avail',
    # 5. Evidence Availability & Missingness (6)
    'both_name_present', 'both_addr_present', 's1_addr_missing', 'cand_addr_missing',
    'both_addr_missing', 'same_country',
    # 6. Interaction Terms & Contradiction Penalties (13)
    'name_addr_prod_tsort', 'name_addr_prod_tset', 'name_addr_prod_lev',
    'name_addr_min_tsort', 'name_addr_min_tset', 'name_addr_max_tsort',
    'name_addr_max_tset', 'name_addr_gap', 'name_with_no_addr',
    'comb_tsort', 'comb_tset', 'comb_lev', 'contradiction_flag'
]

# ── Normalization Regexes ─────────────────────────────────────────────────────
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

class PhoneticMemoizer:
    def __init__(self):
        self.memo = {}

    def get_keys(self, text: str) -> tuple:
        if not text: return ()
        keys = []
        for tok in str(text).lower().split():
            if len(tok) >= 3:
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

# ── Feature Computer ──────────────────────────────────────────────────────────
def trigram_jaccard(s1: str, s2: str) -> float:
    if not s1 or not s2: return 0.0
    t1 = {s1[i:i+3] for i in range(len(s1)-2)} if len(s1) >= 3 else {s1}
    t2 = {s2[i:i+3] for i in range(len(s2)-2)} if len(s2) >= 3 else {s2}
    u = len(t1 | t2)
    return len(t1 & t2) / u if u > 0 else 0.0

def compute_pair_features(s1_nn, s1_na, s1_country, s1_rn, s1_ra,
                          c_nn, c_na, c_country, c_rn, c_ra,
                          s1_pc, c_pc):
    feats = np.zeros(48, dtype=np.float32)

    # 1. Name Similarities
    if s1_nn and c_nn:
        tok1, tok2 = set(s1_nn.split()), set(c_nn.split())
        u = len(tok1 | tok2)
        feats[0] = len(tok1 & tok2) / u if u > 0 else 0.0
        feats[1] = trigram_jaccard(s1_nn, c_nn)
        feats[2] = fuzz.ratio(s1_nn, c_nn) / 100.0
        feats[3] = fuzz.token_sort_ratio(s1_nn, c_nn) / 100.0
        feats[4] = fuzz.token_set_ratio(s1_nn, c_nn) / 100.0
        feats[5] = fuzz.partial_ratio(s1_nn, c_nn) / 100.0
        feats[6] = fuzz.WRatio(s1_nn, c_nn) / 100.0
        feats[7] = 1.0 if s1_nn == c_nn else 0.0
        feats[8] = 1.0 if feats[3] == 1.0 else 0.0
        feats[9] = abs(len(s1_nn) - len(c_nn))
        feats[10] = min(len(s1_nn), len(c_nn)) / max(len(s1_nn), len(c_nn), 1)
        feats[11] = 1.0 if s1_nn[:3] == c_nn[:3] else 0.0

    # 2. Address Similarities
    both_addr = 1.0 if (s1_na and c_na) else 0.0
    if both_addr:
        tok1, tok2 = set(s1_na.split()), set(c_na.split())
        u = len(tok1 | tok2)
        feats[12] = len(tok1 & tok2) / u if u > 0 else 0.0
        feats[13] = trigram_jaccard(s1_na, c_na)
        feats[14] = fuzz.ratio(s1_na, c_na) / 100.0
        feats[15] = fuzz.token_sort_ratio(s1_na, c_na) / 100.0
        feats[16] = fuzz.token_set_ratio(s1_na, c_na) / 100.0
        feats[17] = fuzz.partial_ratio(s1_na, c_na) / 100.0
        
        n1 = fast_get_nums(s1_na)
        n2 = fast_get_nums(c_na)
        feats[18] = 1.0 if (n1 and n2 and n1[0] == n2[0]) else 0.0
        sn1, sn2 = set(n1), set(n2)
        un = len(sn1 | sn2)
        feats[19] = len(sn1 & sn2) / un if un > 0 else 0.0
        feats[20] = abs(len(s1_na) - len(c_na))
        feats[21] = min(len(s1_na), len(c_na)) / max(len(s1_na), len(c_na), 1)

    # 3. Phonetics
    pk1 = set(memoizer.get_keys(s1_rn))
    pk2 = set(memoizer.get_keys(c_rn))
    if pk1 and pk2:
        feats[22] = len(pk1 & pk2)
        feats[23] = len(pk1 & pk2) / len(pk1 | pk2)

    if s1_ra and c_ra:
        apk1 = set(memoizer.get_keys(s1_ra))
        apk2 = set(memoizer.get_keys(c_ra))
        if apk1 and apk2:
            feats[24] = len(apk1 & apk2)
            feats[25] = len(apk1 & apk2) / len(apk1 | apk2)

    # 4. Postal
    if s1_pc and c_pc:
        feats[26] = 1.0 if s1_pc == c_pc else 0.0
        feats[27] = 1.0 if s1_pc != c_pc else 0.0
        feats[28] = 1.0
    elif s1_pc or c_pc:
        feats[28] = 0.5

    # 5. Missingness
    feats[29] = 1.0 if (s1_nn and c_nn) else 0.0
    feats[30] = both_addr
    feats[31] = 1.0 if (not s1_na and c_na) else 0.0
    feats[32] = 1.0 if (s1_na and not c_na) else 0.0
    feats[33] = 1.0 if (not s1_na and not c_na) else 0.0
    feats[34] = 1.0 if s1_country == c_country else 0.0

    # 6. Cross-Field Interactions & Contradiction Penalties
    n_tsort = feats[3]
    n_tset  = feats[4]
    n_lev   = feats[2]
    a_tsort = feats[15]
    a_tset  = feats[16]
    a_lev   = feats[14]

    feats[35] = n_tsort * a_tsort
    feats[36] = n_tset * a_tset
    feats[37] = n_lev * a_lev
    feats[38] = min(n_tsort, a_tsort)
    feats[39] = min(n_tset, a_tset)
    feats[40] = max(n_tsort, a_tsort)
    feats[41] = max(n_tset, a_tset)
    feats[42] = abs(n_tsort - a_tsort) if both_addr else 0.0
    feats[43] = n_tsort * (1.0 - both_addr)

    if s1_nn and c_nn and (s1_na or c_na):
        c1 = f"{s1_nn} {s1_na}".strip()
        c2 = f"{c_nn} {c_na}".strip()
        feats[44] = fuzz.token_sort_ratio(c1, c2) / 100.0
        feats[45] = fuzz.token_set_ratio(c1, c2) / 100.0
        feats[46] = fuzz.ratio(c1, c2) / 100.0

    # Contradiction flag
    if both_addr and len(s1_na) >= 8 and len(c_na) >= 8:
        if a_tset < 0.20 and (feats[27] == 1.0 or feats[18] == 0.0):
            feats[47] = 1.0

    return feats

# ── Production Inference & Checkpointing ──────────────────────────────────────
def run_production_inference(test_dir: Path, output_dir: Path, model_path: Path,
                             chunk_size: int = 30000, limit: int = None):
    t_start = time.perf_counter()
    print("=" * 75, flush=True)
    print("PRODUCTION INFERENCE RUNNER — AMAZON ML CHALLENGE 2026", flush=True)
    print(f"Initial RAM RSS: {get_ram_mb():.1f} MB", flush=True)
    print("=" * 75, flush=True)

    matching_out = output_dir / "matching_results.tsv"
    candidate_out = output_dir / "candidate_pairs.tsv"
    checkpoint_file = output_dir / ".inference_checkpoint.json"

    # 1. Load Model
    t_m0 = time.perf_counter()
    print(f"Loading LightGBM Model from {model_path}...", flush=True)
    model = lgb.Booster(model_file=str(model_path))
    t_model_load = time.perf_counter() - t_m0
    print(f"  Model loaded in {t_model_load:.3f}s (RAM: {get_ram_mb():.1f} MB).", flush=True)

    # 2. Candidate Index & Normalization Setup (Cache or Build)
    t_cache_load = 0.0
    t_cand_load = 0.0
    t_norm_total = 0.0
    t_idx_total = 0.0

    cache_file = cache_dir / "test_candidate_index.pkl"
    if cache_file.exists():
        print(f"\n[2/4] Loading candidate pool & 8 inverted indices from cache: {cache_file}...", flush=True)
        t_c0 = time.perf_counter()
        with open(cache_file, "rb") as f:
            cache_payload = pickle.load(f)
        n_cands = cache_payload["n_cands"]
        c_ids = cache_payload["c_ids"]
        c_raw_names = cache_payload["c_raw_names"]
        c_raw_addrs = cache_payload["c_raw_addrs"]
        c_countries = cache_payload["c_countries"]
        c_nn = cache_payload["c_nn"]
        c_nns = cache_payload["c_nns"]
        c_na = cache_payload["c_na"]
        c_pc = cache_payload["c_pc"]
        idx_exact = cache_payload["idx_exact"]
        idx_pfx3 = cache_payload["idx_pfx3"]
        idx_phone = cache_payload["idx_phone"]
        idx_postal = cache_payload["idx_postal"]
        idx_addr_key = cache_payload["idx_addr_key"]
        idx_comp = cache_payload["idx_comp"]
        idx_rare_tok = cache_payload["idx_rare_tok"]
        idx_rare_addr = cache_payload["idx_rare_addr"]
        if "memoizer_memo" in cache_payload:
            memoizer.memo.update(cache_payload["memoizer_memo"])
        del cache_payload
        t_cache_load = time.perf_counter() - t_c0
        print(f"  Loaded {n_cands:,} candidate entities & 8 indices in {t_cache_load:.1f}s (RAM: {get_ram_mb():.1f} MB).", flush=True)
    else:
        # Load Candidate Datasets (Test S2 + S3)
        s2_path = test_dir / "test_source2.tsv"
        s3_path = test_dir / "test_source3.tsv"

        print(f"\n[1/4] Loading candidate sources...", flush=True)
        t0 = time.perf_counter()
        s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
        t_s2 = time.perf_counter() - t0
        print(f"  Loaded {len(s2):,} S2 rows ({t_s2:.1f}s, RAM: {get_ram_mb():.1f} MB).", flush=True)

        t0 = time.perf_counter()
        s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
        t_s3 = time.perf_counter() - t0
        print(f"  Loaded {len(s3):,} S3 rows ({t_s3:.1f}s, RAM: {get_ram_mb():.1f} MB).", flush=True)

        cand_df = pd.concat([s2, s3], ignore_index=True)
        del s2, s3
        gc.collect()
        n_cands = len(cand_df)
        print(f"  Candidate pool size: {n_cands:,} entities (RAM: {get_ram_mb():.1f} MB).", flush=True)

        # Normalization & Inverted Indexing
        print(f"\n[2/4] High-speed normalization & token frequency profiling...", flush=True)
        t_norm_start = time.perf_counter()
        c_ids = cand_df["entity_id"].values
        c_raw_names = cand_df["business_name"].values
        c_raw_addrs = cand_df["business_address"].values
        c_countries = cand_df["country"].values

        c_nn = [fast_norm_name(x) for x in c_raw_names]
        c_nns = [fast_norm_name_sorted(nn) for nn in c_nn]
        c_na = [fast_norm_addr(x) for x in c_raw_addrs]
        c_pc = [fast_get_postal(x) for x in c_raw_addrs]
        t_norm_total = time.perf_counter() - t_norm_start
        print(f"  Normalized {n_cands:,} candidates in {t_norm_total:.1f}s (RAM: {get_ram_mb():.1f} MB).", flush=True)

        print(f"\n[3/4] Building 8 inverted blocking indices (single pass)...", flush=True)
        t_idx_start = time.perf_counter()
        idx_exact = defaultdict(list)
        idx_pfx3 = defaultdict(list)
        idx_phone = defaultdict(list)
        idx_postal = defaultdict(list)
        idx_addr_key = defaultdict(list)
        idx_comp = defaultdict(list)
        idx_rare_tok = defaultdict(list)
        idx_rare_addr = defaultdict(list)

        for i in range(n_cands):
            nns_v = c_nns[i]
            if nns_v: idx_exact[nns_v].append(i)
            
            nn_v = c_nn[i]
            if len(nn_v) >= 3:
                idx_pfx3[nn_v[:3]].append(i)
                # Rare name tokens
                for w in nn_v.split():
                    if len(w) >= 3:
                        idx_rare_tok[w].append(i)

            for pk in memoizer.get_keys(c_raw_names[i]):
                idx_phone[pk].append(i)

            pc_v = c_pc[i]
            if pc_v: idx_postal[pc_v].append(i)

            na_v = c_na[i]
            if na_v:
                for ak in get_addr_keys(na_v):
                    idx_addr_key[ak].append(i)
                for w in na_v.split():
                    if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS:
                        idx_rare_addr[w].append(i)

            for ck in get_name_addr_composite(nn_v, na_v):
                idx_comp[ck].append(i)

        # In-place pruning of high-frequency hub tokens
        for k in list(idx_rare_tok.keys()):
            if len(idx_rare_tok[k]) > 10000:
                del idx_rare_tok[k]

        for k in list(idx_rare_addr.keys()):
            if len(idx_rare_addr[k]) > 600:
                del idx_rare_addr[k]

        del cand_df
        gc.collect()
        t_idx_total = time.perf_counter() - t_idx_start
        print(f"  8 inverted indices built in {t_idx_total:.1f}s (RAM: {get_ram_mb():.1f} MB).", flush=True)

    # 4. Checkpoint Resumption
    processed_s1_count = 0
    if limit is None and checkpoint_file.exists() and matching_out.exists() and candidate_out.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                processed_s1_count = ckpt.get("processed_s1", 0)
                print(f"[Checkpoint] Resuming from S1 entity offset: {processed_s1_count:,}", flush=True)
        except Exception:
            processed_s1_count = 0

    mode = "a" if (limit is None and processed_s1_count > 0) else "w"
    s1_path = test_dir / "test_source1.tsv"
    effective_chunk = min(chunk_size, limit) if limit else chunk_size
    print(f"\n[4/4] Streaming {s1_path} in chunks of {effective_chunk:,} (mode='{mode}')...", flush=True)

    s1_reader = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False,
                            encoding="utf-8", chunksize=effective_chunk)

    total_s1 = 0
    total_cand_pairs = 0
    total_matches = 0
    total_singletons = 0

    cum_cand_gen_time = 0.0
    cum_feat_comp_time = 0.0
    cum_score_time = 0.0
    cum_write_time = 0.0

    with open(matching_out, mode, encoding="utf-8", newline="\n") as f_match, \
         open(candidate_out, mode, encoding="utf-8", newline="\n") as f_cand:

        if processed_s1_count == 0 or limit is not None:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        batch_idx = 0
        current_offset = 0

        for s1_chunk in s1_reader:
            batch_idx += 1
            n_chunk = len(s1_chunk)
            current_offset += n_chunk

            if limit is None and current_offset <= processed_s1_count:
                print(f"  Skipping already processed batch {batch_idx:3d} ({current_offset:,} S1 entities)...", flush=True)
                total_s1 = current_offset
                continue

            b_start = time.perf_counter()
            s1_ids = s1_chunk["entity_id"].values
            s1_raw_names = s1_chunk["business_name"].values
            s1_raw_addrs = s1_chunk["business_address"].values
            s1_countries = s1_chunk["country"].values

            s1_nn = [fast_norm_name(x) for x in s1_raw_names]
            s1_nns = [fast_norm_name_sorted(nn) for nn in s1_nn]
            s1_na = [fast_norm_addr(x) for x in s1_raw_addrs]
            s1_pc = [fast_get_postal(x) for x in s1_raw_addrs]

            # Step A: Candidate Generation (Blocking)
            t_cg0 = time.perf_counter()
            batch_pairs = []
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

                if nn_v:
                    for w in nn_v.split():
                        if len(w) >= 3:
                            b = idx_rare_tok.get(w, [])
                            if len(b) <= 800: s_cands.update(b)

                pc_v = s1_pc[i]
                if pc_v:
                    s_cands.update(idx_postal.get(pc_v, []))

                na_v = s1_na[i]
                if na_v:
                    for ak in get_addr_keys(na_v):
                        b = idx_addr_key.get(ak, [])
                        if len(b) <= 400: s_cands.update(b)
                    for w in na_v.split():
                        if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS:
                            b = idx_rare_addr.get(w, [])
                            if len(b) <= 400: s_cands.update(b)

                for ck in get_name_addr_composite(nn_v, na_v):
                    b = idx_comp.get(ck, [])
                    if len(b) <= 400: s_cands.update(b)

                if len(s_cands) > 50:
                    c_list = list(s_cands)[:50]
                else:
                    c_list = list(s_cands)

                s1_cand_sets.append({c_ids[j] for j in c_list})
                for j in c_list:
                    batch_pairs.append((i, j))

            t_cg = time.perf_counter() - t_cg0
            cum_cand_gen_time += t_cg

            n_pairs = len(batch_pairs)
            total_cand_pairs += n_pairs
            s1_matches = [[] for _ in range(n_chunk)]

            # Step B: Feature Extraction
            t_fc0 = time.perf_counter()
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
                    pair_meta.append((s1_i, c_ids[cand_j], feat[30], feat[32], feat[3], feat[16], feat[47]))

                t_fc = time.perf_counter() - t_fc0
                cum_feat_comp_time += t_fc

                # Step C: Model Scoring & Decision Rules
                t_sc0 = time.perf_counter()
                probs = model.predict(X_batch)
                del X_batch

                # Calibrated Tiered Decision Rules (EXP-04)
                s1_scores = defaultdict(list)
                for p_idx, p_score in enumerate(probs):
                    s1_i, cid, both_a, cand_miss, n_tsort, a_tset, contra = pair_meta[p_idx]

                    # 1. Contradiction Veto Filter
                    if contra >= 0.80 and p_score < 0.995:
                        continue

                    # 2. Tiered Evidence Thresholds
                    if both_a == 1.0 and n_tsort >= 0.80 and a_tset >= 0.80:
                        eff_thr = 0.85
                    elif cand_miss == 1.0 or both_a == 0.0:
                        eff_thr = 0.99
                    else:
                        eff_thr = 0.97

                    if p_score >= eff_thr:
                        s1_scores[s1_i].append((cid, p_score))

                # 3. Margin Pruning
                for s1_i, scored_cands in s1_scores.items():
                    if not scored_cands:
                        continue
                    max_p = max(p for _, p in scored_cands)
                    accepted = [cid for cid, p in scored_cands if p >= (max_p - 0.15)]
                    s1_matches[s1_i] = accepted
                    total_matches += len(accepted)

                t_sc = time.perf_counter() - t_sc0
                cum_score_time += t_sc
            else:
                t_fc = 0.0
                t_sc = 0.0

            # Step D: Output Writing (with matched ⊆ candidates guarantee)
            t_w0 = time.perf_counter()
            for i in range(n_chunk):
                sid = s1_ids[i]
                c_set = s1_cand_sets[i]
                m_list = s1_matches[i]

                if not m_list:
                    total_singletons += 1

                for mid in m_list:
                    c_set.add(mid)

                m_str = ",".join(m_list)
                c_str = ",".join(sorted(c_set))

                f_match.write(f"{sid}\t{m_str}\n")
                f_cand.write(f"{sid}\t{c_str}\n")

            f_match.flush()
            f_cand.flush()
            t_w = time.perf_counter() - t_w0
            cum_write_time += t_w

            # Memory cleanup
            del batch_pairs, s1_cand_sets, s1_matches
            if n_pairs > 0:
                del pair_meta, s1_scores

            b_total = time.perf_counter() - b_start
            total_s1 += n_chunk
            print(f"  Batch {batch_idx:3d}: Processed {total_s1:,} S1 entities "
                  f"({n_pairs:,} pairs scored, {b_total:.2f}s | "
                  f"CandGen: {t_cg:.2f}s, Feat: {t_fc:.2f}s, Score: {t_sc:.2f}s, Write: {t_w:.2f}s | "
                  f"RAM: {get_ram_mb():.1f} MB)", flush=True)

            if limit is None:
                with open(checkpoint_file, "w", encoding="utf-8") as f_ckpt:
                    json.dump({"processed_s1": total_s1, "timestamp": time.time()}, f_ckpt)

            if limit and total_s1 >= limit:
                print(f"  Reached limit of {limit:,} entities. Benchmark complete.", flush=True)
                break

    t_total_run = time.perf_counter() - t_start
    print("\n" + "=" * 75, flush=True)
    print("DETAILED EXECUTION & TIMING REPORT", flush=True)
    print("=" * 75, flush=True)
    print(f"  Startup & Model Load Time:      {t_model_load:.3f} s")
    if t_cache_load > 0:
        print(f"  Candidate Index Cache Load Time:{t_cache_load:.3f} s")
    else:
        print(f"  Candidate Loading Time:         {t_cand_load:.3f} s")
        print(f"  Candidate Normalization Time:   {t_norm_total:.3f} s")
        print(f"  Index Construction Time:        {t_idx_total:.3f} s")
    print(f"  Candidate Generation Time:      {cum_cand_gen_time:.3f} s")
    print(f"  Feature Computation Time:       {cum_feat_comp_time:.3f} s")
    print(f"  Model Scoring & Decisions Time: {cum_score_time:.3f} s")
    print(f"  Output Writing Time:            {cum_write_time:.3f} s")
    print(f"  Total Elapsed Runtime:          {t_total_run:.3f} s ({(t_total_run/60):.2f} min)")
    print(f"  Peak / Current RAM Usage:       {get_ram_mb():.1f} MB")
    print(f"  Total S1 Entities Processed:    {total_s1:,}")
    print(f"  Total Candidate Pairs Scored:   {total_cand_pairs:,} ({total_cand_pairs/max(total_s1,1):.1f} cands/S1)")
    print(f"  Total Predicted Matches:        {total_matches:,}")
    print(f"  Total Empty / Singletons:       {total_singletons:,} ({100*total_singletons/max(total_s1,1):.1f}%)")
    print(f"  Throughput Speed:               {total_s1/max(cum_cand_gen_time+cum_feat_comp_time+cum_score_time+cum_write_time, 0.001):.1f} S1 entities/sec")
    print(f"  Matching Output Path:           {matching_out}")
    print(f"  Candidate Output Path:          {candidate_out}")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    train_dir, test_dir, output_dir, cache_dir, model_dir = resolve_paths()
    model_path = model_dir / "final_model.txt"
    if not model_path.exists():
        model_path = cache_dir / "exp03_model.txt"
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_production_inference(test_dir, output_dir, model_path,
                             chunk_size=args.chunk_size, limit=args.limit)
    os._exit(0)
