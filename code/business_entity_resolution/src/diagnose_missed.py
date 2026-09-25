#!/usr/bin/env python3
"""
Diagnostic: Inspect Missed True Match Pairs to identify exact failure patterns.
"""
import os, sys, time, gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp02b_blocking import (pkeys, get_rare_tokens, get_addr_keys,
                             get_name_addr_composite, DEV_S1_SAMPLE, CAND_NEG_SAMPLE)
from core import load_source, load_ground_truth, add_norm_cols

def main():
    s1_full = load_source("train", 1)
    s2_full = load_source("train", 2)
    s3_full = load_source("train", 3)
    gt_full = load_ground_truth()

    rng = np.random.RandomState(42)
    all_ids = list(gt_full.keys())
    rng.shuffle(all_ids)
    dev_ids = all_ids[:5000] # 5k for quick inspection
    
    true_match_ids = set()
    for sid in dev_ids:
        true_match_ids.update(gt_full[sid])
        
    s1 = s1_full[s1_full["entity_id"].isin(dev_ids)].copy().reset_index(drop=True)
    s2 = s2_full[s2_full["entity_id"].isin(true_match_ids)].copy().reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(true_match_ids)].copy().reset_index(drop=True)
    gt_dev = {k: gt_full[k] for k in dev_ids}
    
    add_norm_cols(s1); add_norm_cols(s2); add_norm_cols(s3)
    cdf = pd.concat([s2, s3], ignore_index=True)
    
    # Build lookups
    s1L = {r["entity_id"]: r for _, r in s1.iterrows()}
    cL  = {r["entity_id"]: r for _, r in cdf.iterrows()}
    
    # Check missed pairs
    cids = cdf["entity_id"].values
    cand_id_to_idx = {cid: i for i, cid in enumerate(cids)}
    
    idx_exact = defaultdict(list)
    for i, v in enumerate(cdf["_nns"].values):
        if v: idx_exact[v].append(i)
        
    idx_pfx3 = defaultdict(list)
    for i, v in enumerate(cdf["_nn"].values):
        if len(v) >= 3: idx_pfx3[v[:3]].append(i)
        
    idx_phone = defaultdict(list)
    for i, rn in enumerate(cdf["business_name"].values):
        for k in pkeys(rn): idx_phone[k].append(i)
        
    idx_rare_tok = defaultdict(list)
    for i, nn in enumerate(cdf["_nn"].values):
        for w in get_rare_tokens(nn): idx_rare_tok[w].append(i)
        
    idx_postal = defaultdict(list)
    for i, pc in enumerate(cdf["_pc"].values):
        if pc: idx_postal[pc].append(i)
        
    idx_addr_key = defaultdict(list)
    for i, na in enumerate(cdf["_na"].values):
        for ak in get_addr_keys(na): idx_addr_key[ak].append(i)
        
    idx_comp = defaultdict(list)
    for i, (nn, na) in enumerate(zip(cdf["_nn"].values, cdf["_na"].values)):
        for ck in get_name_addr_composite(nn, na): idx_comp[ck].append(i)

    print("\n=== SAMPLE OF MISSED TRUE MATCH PAIRS ===", flush=True)
    missed_count = 0
    patterns = Counter()
    
    for sid, tm in gt_dev.items():
        if not tm: continue
        s_row = s1L[sid]
        nn = s_row["_nn"]
        nns = s_row["_nns"]
        rn = s_row["business_name"]
        pc = s_row["_pc"]
        na = s_row["_na"]
        
        s_union = set(idx_exact.get(nns, []))
        pfx3 = nn[:3] if len(nn) >= 3 else nn
        if pfx3: s_union.update(idx_pfx3.get(pfx3, []))
        for k in pkeys(rn): s_union.update(idx_phone.get(k, []))
        for w in get_rare_tokens(nn): s_union.update(idx_rare_tok.get(w, []))
        if pc: s_union.update(idx_postal.get(pc, []))
        for ak in get_addr_keys(na): s_union.update(idx_addr_key.get(ak, []))
        for ck in get_name_addr_composite(nn, na): s_union.update(idx_comp.get(ck, []))
        
        union_ids = {cids[j] for j in s_union}
        missed = tm - union_ids
        
        for mid in missed:
            if mid not in cL: continue
            c_row = cL[mid]
            missed_count += 1
            
            # Diagnose reason
            s_name, c_name = s_row["business_name"], c_row["business_name"]
            s_addr, c_addr = s_row["business_address"], c_row["business_address"]
            
            if missed_count <= 15:
                print(f"\n[Missed #{missed_count}] S1: {sid} vs Match: {mid}")
                print(f"  S1 Name : {s_name}")
                print(f"  Cand Name: {c_name}")
                print(f"  S1 Addr : {s_addr}")
                print(f"  Cand Addr: {c_addr}")
                
            if s_name.lower().replace(" ", "") == c_name.lower().replace(" ", ""):
                patterns["whitespace_diff"] += 1
            elif any(w in c_name.lower() for w in s_name.lower().split() if len(w) >= 3):
                patterns["partial_word_overlap"] += 1
            elif s_addr and c_addr and (s_addr.split(",")[0].strip().lower() == c_addr.split(",")[0].strip().lower()):
                patterns["same_street_diff_name"] += 1
            else:
                patterns["other_noisy"] += 1
                
    print(f"\nTotal Missed in Sample: {missed_count}", flush=True)
    print("Missed Patterns:", patterns, flush=True)

if __name__ == "__main__":
    main()
