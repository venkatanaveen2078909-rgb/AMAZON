#!/usr/bin/env python3
r"""
Precomputes and caches normalized candidate arrays and 8 inverted blocking indices
to disk for instant reuse across all subsequent benchmark runs and production inference.
"""
import os
import sys
import gc
import re
import time
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd
from metaphone import doublemetaphone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Paths
def resolve_paths():
    curr = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    root = curr
    for _ in range(5):
        if (root / "6ab10eb3b23ba_student_resource").exists():
            break
        root = root.parent
    
    base_res = root / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    if base_res.exists():
        test_dir = base_res / "test"
    else:
        test_dir = root / "dataset" / "test"
        
    cache_dir = root / "code" / "business_entity_resolution" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "test_candidate_index.pkl"
    return test_dir, cache_dir, cache_file

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

def main():
    test_dir, cache_dir, cache_file = resolve_paths()
    t_start = time.perf_counter()
    print("=" * 75, flush=True)
    print("BUILDING REUSABLE CANDIDATE INDEX & NORMALIZATION CACHE", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Candidate Datasets
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print(f"Loading {s2_path} and {s3_path}...", flush=True)
    t0 = time.perf_counter()
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    cand_df = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    n_cands = len(cand_df)
    print(f"Loaded {n_cands:,} candidate entities in {time.perf_counter()-t0:.1f}s.", flush=True)

    # 2. Normalization
    print(f"Normalizing candidate entities...", flush=True)
    t_norm = time.perf_counter()
    c_ids = list(cand_df["entity_id"].values)
    c_raw_names = list(cand_df["business_name"].values)
    c_raw_addrs = list(cand_df["business_address"].values)
    c_countries = list(cand_df["country"].values)

    c_nn = [fast_norm_name(x) for x in c_raw_names]
    c_nns = [fast_norm_name_sorted(nn) for nn in c_nn]
    c_na = [fast_norm_addr(x) for x in c_raw_addrs]
    c_pc = [fast_get_postal(x) for x in c_raw_addrs]
    print(f"Normalized in {time.perf_counter()-t_norm:.1f}s.", flush=True)

    # 3. Build 8 Inverted Indices (Single Pass)
    print(f"Building 8 inverted blocking indices...", flush=True)
    t_idx = time.perf_counter()
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

    # Prune hub keys in-place
    print("Pruning broad hub keys in-place...", flush=True)
    for k in list(idx_rare_tok.keys()):
        if len(idx_rare_tok[k]) > 10000:
            del idx_rare_tok[k]

    for k in list(idx_rare_addr.keys()):
        if len(idx_rare_addr[k]) > 600:
            del idx_rare_addr[k]

    # Convert defaultdict to normal dict to save memory
    idx_exact = dict(idx_exact)
    idx_pfx3 = dict(idx_pfx3)
    idx_phone = dict(idx_phone)
    idx_postal = dict(idx_postal)
    idx_addr_key = dict(idx_addr_key)
    idx_comp = dict(idx_comp)
    idx_rare_tok = dict(idx_rare_tok)
    idx_rare_addr = dict(idx_rare_addr)

    del cand_df
    gc.collect()
    print(f"Indices constructed in {time.perf_counter()-t_idx:.1f}s.", flush=True)

    # 4. Save Cache
    print(f"Saving serialized cache to {cache_file}...", flush=True)
    t_save = time.perf_counter()
    cache_payload = {
        "n_cands": n_cands,
        "c_ids": c_ids,
        "c_raw_names": c_raw_names,
        "c_raw_addrs": c_raw_addrs,
        "c_countries": c_countries,
        "c_nn": c_nn,
        "c_nns": c_nns,
        "c_na": c_na,
        "c_pc": c_pc,
        "idx_exact": idx_exact,
        "idx_pfx3": idx_pfx3,
        "idx_phone": idx_phone,
        "idx_postal": idx_postal,
        "idx_addr_key": idx_addr_key,
        "idx_comp": idx_comp,
        "idx_rare_tok": idx_rare_tok,
        "idx_rare_addr": idx_rare_addr,
        "memoizer_memo": memoizer.memo
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"Saved cache file ({cache_size_mb:.1f} MB) in {time.perf_counter()-t_save:.1f}s.", flush=True)
    print(f"Total build time: {time.perf_counter()-t_start:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
    os._exit(0)
