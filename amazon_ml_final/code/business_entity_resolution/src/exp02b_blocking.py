#!/usr/bin/env python3
"""
Experiment 02b: Maximizing Blocking Recall to >98%.
Adds:
  7. Character 3-Gram Inverted Index (Name)
  8. Address Street Number + City token
  9. Name 2-char prefix + Address number
"""
import os, sys, time, gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from metaphone import doublemetaphone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_source, load_ground_truth, add_norm_cols,
                  norm_name, norm_addr, get_postal, get_nums,
                  blocking_recall, log_experiment)

DEV_S1_SAMPLE   = 20000
CAND_NEG_SAMPLE = 300000

def pkeys(name: str) -> set:
    n = norm_name(name)
    if not n: return set()
    res = set()
    for tok in n.split():
        if len(tok) >= 2:
            try:
                p, s = doublemetaphone(tok)
                if p: res.add(p)
                if s: res.add(s)
            except Exception:
                pass
    return res

def get_rare_tokens(norm_name_str: str) -> set:
    if not norm_name_str: return set()
    return {w for w in norm_name_str.split() if len(w) >= 3}

def get_trigrams(s: str) -> set:
    if not s or len(s) < 3: return set()
    return {s[i:i+3] for i in range(len(s)-2)}

def get_addr_keys(na_str: str) -> set:
    if not na_str: return set()
    nums = get_nums(na_str)
    toks = [w for w in na_str.split() if len(w) >= 3 and not w.isdigit()]
    keys = set()
    if nums and toks:
        keys.add(f"{nums[0]}_{toks[0]}")
        if len(toks) > 1:
            keys.add(f"{nums[0]}_{toks[-1]}")
    return keys

def get_name_addr_composite(nn_str: str, na_str: str) -> set:
    if not nn_str or not na_str: return set()
    pfx2 = nn_str[:2]
    nums = get_nums(na_str)
    if pfx2 and nums:
        return {f"{pfx2}_{nums[0]}"}
    return set()

def main():
    print("=" * 75, flush=True)
    print("EXPERIMENT 02b: HIGH-RECALL MULTI-VIEW BLOCKING (>98% TARGET)", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Data
    s1_full = load_source("train", 1)
    s2_full = load_source("train", 2)
    s3_full = load_source("train", 3)
    gt_full = load_ground_truth()

    rng = np.random.RandomState(42)
    all_ids = list(gt_full.keys())
    rng.shuffle(all_ids)
    dev_ids = all_ids[:DEV_S1_SAMPLE]
    
    true_match_ids = set()
    for sid in dev_ids:
        true_match_ids.update(gt_full[sid])
    print(f"Dev S1: {len(dev_ids):,}, True matches needed: {len(true_match_ids):,}", flush=True)
    
    all_cand_ids = set(s2_full["entity_id"]) | set(s3_full["entity_id"])
    neg_pool = list(all_cand_ids - true_match_ids)
    rng.shuffle(neg_pool)
    neg_sample = set(neg_pool[:CAND_NEG_SAMPLE])
    keep_cand_ids = true_match_ids | neg_sample

    s1 = s1_full[s1_full["entity_id"].isin(dev_ids)].copy().reset_index(drop=True)
    s2 = s2_full[s2_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    gt_dev = {k: gt_full[k] for k in dev_ids}
    
    del s1_full, s2_full, s3_full, gt_full
    gc.collect()

    add_norm_cols(s1); add_norm_cols(s2); add_norm_cols(s3)
    
    cdf = pd.concat([s2, s3], ignore_index=True)
    n = len(cdf)
    cids = cdf["entity_id"].values
    nns_v = cdf["_nns"].values
    nn_v  = cdf["_nn"].values
    pc_v  = cdf["_pc"].values
    na_v  = cdf["_na"].values
    rn_v  = cdf["business_name"].values

    print(f"\nBuilding comprehensive indices on {n:,} candidates...", flush=True)
    
    # 1. Exact Name
    idx_exact = defaultdict(list)
    for i, v in enumerate(nns_v):
        if v: idx_exact[v].append(i)
    idx_exact = dict(idx_exact)
    
    # 2. Prefix-3 & Prefix-4
    idx_pfx3 = defaultdict(list)
    for i, v in enumerate(nn_v):
        if len(v) >= 3: idx_pfx3[v[:3]].append(i)
    idx_pfx3 = dict(idx_pfx3)
    
    # 3. Phonetic
    idx_phone = defaultdict(list)
    for i in range(n):
        for k in pkeys(rn_v[i]):
            idx_phone[k].append(i)
    idx_phone = dict(idx_phone)
    
    # 4. Rare Name Word Tokens
    tok_df = Counter()
    for i in range(n):
        for w in get_rare_tokens(nn_v[i]):
            tok_df[w] += 1
    
    idx_rare_tok = defaultdict(list)
    for i in range(n):
        for w in get_rare_tokens(nn_v[i]):
            if tok_df[w] < 15000:
                idx_rare_tok[w].append(i)
    idx_rare_tok = dict(idx_rare_tok)
    
    # 5. Postal
    idx_postal = defaultdict(list)
    for i, v in enumerate(pc_v):
        if v: idx_postal[v].append(i)
    idx_postal = dict(idx_postal)
    
    # 6. Addr Key (Num + Street word / City word)
    idx_addr_key = defaultdict(list)
    for i in range(n):
        for ak in get_addr_keys(na_v[i]):
            idx_addr_key[ak].append(i)
    idx_addr_key = dict(idx_addr_key)
    
    # 7. Name 2-char + Address Num Composite Key
    idx_comp = defaultdict(list)
    for i in range(n):
        for ck in get_name_addr_composite(nn_v[i], na_v[i]):
            idx_comp[ck].append(i)
    idx_comp = dict(idx_comp)
    print(f"  Composite Key Index: {len(idx_comp):,} keys", flush=True)

    # 5. Fast Evaluation
    cand_id_to_idx = {cid: i for i, cid in enumerate(cids)}
    gt_int = {}
    tot_true_pairs = 0
    for sid, tm in gt_dev.items():
        tm_idx = {cand_id_to_idx[cid] for cid in tm if cid in cand_id_to_idx}
        gt_int[sid] = tm_idx
        tot_true_pairs += len(tm_idx)
        
    s1_ids = s1["entity_id"].values
    s1_nns = s1["_nns"].values
    s1_nn  = s1["_nn"].values
    s1_rn  = s1["business_name"].values
    s1_pc  = s1["_pc"].values
    s1_na  = s1["_na"].values

    stats = {
        "1. Exact Name": {"found": 0, "pairs": 0},
        "2. Name Prefix-3": {"found": 0, "pairs": 0},
        "3. Phonetic": {"found": 0, "pairs": 0},
        "4. Rare Name Words": {"found": 0, "pairs": 0},
        "5. Postal Code": {"found": 0, "pairs": 0},
        "6. Addr Num+Word": {"found": 0, "pairs": 0},
        "7. Name2+AddrNum": {"found": 0, "pairs": 0},
        "UNION (ALL)": {"found": 0, "pairs": 0},
    }

    t_eval_start = time.time()
    for sid, nn, nns, rn, pc, na in zip(s1_ids, s1_nn, s1_nns, s1_rn, s1_pc, s1_na):
        true_set = gt_int[sid]
        
        # 1. Exact Name
        s_exact = set(idx_exact.get(nns, []))
        stats["1. Exact Name"]["pairs"] += len(s_exact)
        stats["1. Exact Name"]["found"] += len(s_exact & true_set)
        
        # 2. Pfx-3 (limit bucket to <= 800)
        pfx3 = nn[:3] if len(nn) >= 3 else nn
        s_pfx = set(idx_pfx3.get(pfx3, [])) if (pfx3 and len(idx_pfx3.get(pfx3, [])) <= 800) else set()
        stats["2. Name Prefix-3"]["pairs"] += len(s_pfx)
        stats["2. Name Prefix-3"]["found"] += len(s_pfx & true_set)
        
        # 3. Phonetic (limit bucket to <= 800)
        s_phone = set()
        for k in pkeys(rn):
            b = idx_phone.get(k, [])
            if len(b) <= 800: s_phone.update(b)
        stats["3. Phonetic"]["pairs"] += len(s_phone)
        stats["3. Phonetic"]["found"] += len(s_phone & true_set)
        
        # 4. Rare Tokens (distinctive words, bucket <= 800)
        s_rare = set()
        for w in get_rare_tokens(nn):
            b = idx_rare_tok.get(w, [])
            if len(b) <= 800: s_rare.update(b)
        stats["4. Rare Name Words"]["pairs"] += len(s_rare)
        stats["4. Rare Name Words"]["found"] += len(s_rare & true_set)
        
        # 5. Postal
        s_post = set(idx_postal.get(pc, [])) if pc else set()
        stats["5. Postal Code"]["pairs"] += len(s_post)
        stats["5. Postal Code"]["found"] += len(s_post & true_set)
        
        # 6. Addr Key (bucket <= 400)
        s_ak = set()
        for ak in get_addr_keys(na):
            b = idx_addr_key.get(ak, [])
            if len(b) <= 400: s_ak.update(b)
        stats["6. Addr Num+Word"]["pairs"] += len(s_ak)
        stats["6. Addr Num+Word"]["found"] += len(s_ak & true_set)
        
        # 7. Name2 + Addr Num (bucket <= 400)
        s_comp = set()
        for ck in get_name_addr_composite(nn, na):
            b = idx_comp.get(ck, [])
            if len(b) <= 400: s_comp.update(b)
        stats["7. Name2+AddrNum"]["pairs"] += len(s_comp)
        stats["7. Name2+AddrNum"]["found"] += len(s_comp & true_set)
        
        # Union
        s_union = s_exact | s_pfx | s_phone | s_rare | s_post | s_ak | s_comp
        stats["UNION (ALL)"]["pairs"] += len(s_union)
        stats["UNION (ALL)"]["found"] += len(s_union & true_set)

    print("\n" + "=" * 75, flush=True)
    print("COMPREHENSIVE BLOCKING RECALL BREAKDOWN", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Strategy':<24} {'Recall Ceiling':>15} {'Total Pairs':>14} {'Avg Cands/S1':>12} {'Missed':>8}", flush=True)
    print("-" * 75, flush=True)
    
    for name, st in stats.items():
        rec = st["found"] / max(tot_true_pairs, 1)
        tot_p = st["pairs"]
        avg_c = tot_p / max(len(s1_ids), 1)
        miss = tot_true_pairs - st["found"]
        print(f"{name:<24} {rec:>15.4%} {tot_p:>14,} {avg_c:>12.1f} {miss:>8,}", flush=True)

    u_found = stats["UNION (ALL)"]["found"]
    u_rec = u_found / max(tot_true_pairs, 1)
    u_pairs = stats["UNION (ALL)"]["pairs"]
    print("\n" + "=" * 75, flush=True)
    print(f"FINAL UNION RECALL CEILING: {u_rec:.4%}", flush=True)
    print(f"True Pairs: {tot_true_pairs:,}, Found: {u_found:,}, Missed: {tot_true_pairs-u_found:,}", flush=True)
    print(f"Total Candidate Pairs: {u_pairs:,} (Avg {u_pairs/len(s1_ids):.1f} candidates per S1)", flush=True)
    print("=" * 75, flush=True)

    log_experiment({
        "experiment_id": "EXP-02B-BLOCKING",
        "date": "2026-09-25",
        "description": "7-channel high-recall union blocking (Composite + AddrNumWord + RareToken + Prefix3 + Phone)",
        "blocking_recall": f"{u_rec:.4f}",
        "val_entities": len(s1_ids),
        "val_pairs_scored": u_pairs,
        "notes": f"Avg candidates/S1: {u_pairs/len(s1_ids):.1f}, Missed: {tot_true_pairs-u_found}"
    })

if __name__ == "__main__":
    main()
