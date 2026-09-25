#!/usr/bin/env python3
"""
Experiment 01: Baseline blocking — recall ceiling measurement.

Goal: Measure how many true matches survive each blocking strategy
      on a development sample, BEFORE building any classifier.

Blocking strategies tested independently + union:
  A. Exact normalized-sorted name
  B. Name prefix (first 4 chars of normalized name)
  C. Phonetic (Double Metaphone on name tokens)
  D. Postal code exact match
  E. Address number overlap (≥2 shared numeric tokens)

We measure: candidate_recall, candidate_count, runtime for each.
"""
import os, sys, time
import numpy as np
import pandas as pd
from collections import defaultdict
from metaphone import doublemetaphone
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_source, load_ground_truth, add_norm_cols,
                  norm_name, get_postal, get_nums,
                  blocking_recall, entity_split, print_split_stats)

# ── Config ─────────────────────────────────────────────────────────────────
DEV_SAMPLE = 20000  # S1 entities for this experiment (fast iteration)

def phonetic_keys(name):
    n = norm_name(name)
    if not n: return set()
    keys = set()
    for tok in n.split():
        if len(tok) < 2: continue
        try:
            p, s = doublemetaphone(tok)
            if p: keys.add(p)
            if s: keys.add(s)
        except: pass
    return keys

# ── Blocking strategies ────────────────────────────────────────────────────

def build_indices(cands_df):
    """Build all blocking indices from S2+S3 candidates."""
    n = len(cands_df)
    ids = cands_df["entity_id"].values
    nns_v = cands_df["_nns"].values
    nn_v  = cands_df["_nn"].values
    na_v  = cands_df["_na"].values
    pc_v  = cands_df["_pc"].values
    raw_n = cands_df["business_name"].values
    raw_a = cands_df["business_address"].values

    print(f"  Building indices on {n:,} candidates...")

    # A: exact normalized-sorted name
    t0 = time.time()
    idx_name = defaultdict(list)
    for i, v in enumerate(nns_v):
        if v: idx_name[v].append(i)
    print(f"    [A] Name exact:  {len(idx_name):,} keys ({time.time()-t0:.1f}s)")

    # B: name prefix (first 4 chars)
    t0 = time.time()
    idx_pfx = defaultdict(list)
    for i, v in enumerate(nn_v):
        if len(v) >= 4: idx_pfx[v[:4]].append(i)
    print(f"    [B] Name prefix: {len(idx_pfx):,} keys ({time.time()-t0:.1f}s)")

    # C: phonetic
    t0 = time.time()
    idx_phone = defaultdict(list)
    for i in tqdm(range(n), desc="    [C] Phonetic", mininterval=10):
        for k in phonetic_keys(raw_n[i]):
            idx_phone[k].append(i)
    print(f"    [C] Phonetic:    {len(idx_phone):,} keys ({time.time()-t0:.1f}s)")

    # D: postal code
    t0 = time.time()
    idx_postal = defaultdict(list)
    for i, v in enumerate(pc_v):
        if v: idx_postal[v].append(i)
    print(f"    [D] Postal:      {len(idx_postal):,} keys ({time.time()-t0:.1f}s)")

    # E: address numbers (individual number → indices)
    t0 = time.time()
    idx_nums = defaultdict(list)
    for i in range(n):
        for num in get_nums(na_v[i]):
            if len(num) >= 2: idx_nums[num].append(i)
    print(f"    [E] Addr nums:   {len(idx_nums):,} keys ({time.time()-t0:.1f}s)")

    return ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums


def block_strategy_A(nns_val, idx_name):
    """Exact normalized-sorted name."""
    if nns_val and nns_val in idx_name:
        return set(idx_name[nns_val])
    return set()

def block_strategy_B(nn_val, idx_pfx):
    """Name prefix (4 chars)."""
    pfx = nn_val[:4] if len(nn_val) >= 4 else nn_val
    if pfx and pfx in idx_pfx:
        bucket = idx_pfx[pfx]
        return set(bucket) if len(bucket) < 50000 else set()
    return set()

def block_strategy_C(raw_name, idx_phone):
    """Phonetic keys."""
    cands = set()
    for k in phonetic_keys(raw_name):
        bucket = idx_phone.get(k)
        if bucket and len(bucket) < 10000:
            cands.update(bucket)
    return cands

def block_strategy_D(pc_val, idx_postal):
    """Postal code exact match."""
    if pc_val and pc_val in idx_postal:
        return set(idx_postal[pc_val])
    return set()

def block_strategy_E(na_val, idx_nums):
    """Address numbers — require ≥2 shared numeric tokens."""
    nums_list = get_nums(na_val)
    if len(nums_list) < 2: return set()
    hits = defaultdict(int)
    for num in nums_list:
        if len(num) >= 2 and num in idx_nums:
            for idx in idx_nums[num]:
                hits[idx] += 1
    return {idx for idx, cnt in hits.items() if cnt >= 2}


def run_blocking(s1_df, gt_split, cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums):
    """Run each blocking strategy independently + union; report recall."""
    
    strategies = {
        'A_name_exact': lambda row: block_strategy_A(row["_nns"], idx_name),
        'B_name_prefix': lambda row: block_strategy_B(row["_nn"], idx_pfx),
        'C_phonetic':    lambda row: block_strategy_C(row["business_name"], idx_phone),
        'D_postal':      lambda row: block_strategy_D(row["_pc"], idx_postal),
        'E_addr_nums':   lambda row: block_strategy_E(row["_na"], idx_nums),
    }
    
    # Collect candidates per strategy
    results = {name: {} for name in strategies}
    results['UNION'] = {}
    
    s1_ids  = s1_df["entity_id"].values
    
    for i, (_, row) in enumerate(tqdm(s1_df.iterrows(), total=len(s1_df),
                                       desc="  Blocking", mininterval=10)):
        eid = row["entity_id"]
        union = set()
        
        for sname, sfunc in strategies.items():
            idx_set = sfunc(row)
            id_set = {cand_ids[j] for j in idx_set}
            results[sname][eid] = id_set
            union.update(id_set)
        
        results['UNION'][eid] = union
    
    # Measure
    print("\n  === BLOCKING RESULTS ===")
    print(f"  {'Strategy':<18} {'Recall':>8} {'Pairs':>12} {'Avg/S1':>8} {'Incr':>8}")
    print(f"  {'-'*58}")
    
    for sname in list(strategies.keys()) + ['UNION']:
        br = blocking_recall(results[sname], gt_split)
        print(f"  {sname:<18} {br['recall_ceiling']:>8.4f} {br['total_candidates']:>12,} "
              f"{br['avg_cands']:>8.1f} {br['missed_pairs']:>8,}miss")
    
    return results


def main():
    print("=" * 70)
    print("EXPERIMENT 01: BASELINE BLOCKING RECALL")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    s1 = load_source("train", 1)
    s2 = load_source("train", 2)
    s3 = load_source("train", 3)
    gt = load_ground_truth()
    print(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}, GT: {len(gt):,}")
    
    # Normalize
    print("\n[2] Normalizing...")
    add_norm_cols(s1)
    add_norm_cols(s2)
    add_norm_cols(s3)
    
    # Dev sample split
    print(f"\n[3] Dev sample: {DEV_SAMPLE:,} S1 entities")
    gt_train, gt_dev = entity_split(gt, val_frac=0.01, seed=42,
                                     max_val=DEV_SAMPLE)
    # Take exactly DEV_SAMPLE from gt_dev + more if needed
    rng = np.random.RandomState(42)
    all_ids = list(gt.keys())
    rng.shuffle(all_ids)
    dev_ids = set(all_ids[:DEV_SAMPLE])
    gt_dev = {k: gt[k] for k in dev_ids}
    print_split_stats("Dev", gt_dev)
    
    # Build candidate pool
    print("\n[4] Building candidate indices...")
    cands = pd.concat([s2, s3], ignore_index=True)
    print(f"  Candidates: {len(cands):,}")
    cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums = build_indices(cands)
    
    # Run blocking on dev set
    print(f"\n[5] Running blocking on {len(dev_ids):,} dev entities...")
    dev_s1 = s1[s1["entity_id"].isin(dev_ids)].copy()
    results = run_blocking(dev_s1, gt_dev, cand_ids,
                           idx_name, idx_pfx, idx_phone, idx_postal, idx_nums)
    
    # Union recall details
    br = blocking_recall(results['UNION'], gt_dev)
    print(f"\n  UNION BLOCKING RECALL CEILING: {br['recall_ceiling']:.4f}")
    print(f"    True pairs: {br['true_pairs']:,}")
    print(f"    Found:      {br['found_pairs']:,}")
    print(f"    Missed:     {br['missed_pairs']:,}")
    print(f"    Avg cands:  {br['avg_cands']:.1f}")
    
    # Inspect a few missed pairs
    print(f"\n  Sample missed pairs:")
    count = 0
    for s1_id, tm in gt_dev.items():
        if not tm: continue
        missed = tm - results['UNION'].get(s1_id, set())
        if missed and count < 5:
            s1_row = s1[s1["entity_id"] == s1_id].iloc[0]
            print(f"    S1={s1_id}: '{s1_row['business_name']}' @ '{s1_row['business_address']}'")
            for mid in list(missed)[:2]:
                c_row = cands[cands["entity_id"] == mid]
                if len(c_row):
                    c_row = c_row.iloc[0]
                    print(f"      Missed: {mid}: '{c_row['business_name']}' @ '{c_row['business_address']}'")
            count += 1
    
    print("\n  DONE. Next: add TF-IDF blocking channel to improve recall ceiling.")


if __name__ == "__main__":
    main()
