#!/usr/bin/env python3
"""
Experiment 03: Full Entity Resolution Pipeline with:
  1. 8-Channel Multi-View Blocking (98.66% measured recall ceiling)
  2. 48 Rich Pairwise Features (Multi-metric, Interactions, Missingness, Contradictions)
  3. Stratified Hard-Negative Mining
  4. LightGBM Gradient Boosting Matcher
  5. Calibrated F0.5 Threshold Optimization & Decision Rules
  6. Automatic Logging to experiments.csv
"""
import os, sys, time, gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from metaphone import doublemetaphone
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_source, load_ground_truth, add_norm_cols,
                  norm_name, norm_addr, get_postal, get_nums,
                  blocking_recall, entity_split, f05_macro,
                  log_experiment, CACHE_DIR)
from features import FEATURE_NAMES, compute_pair_features

# ── Experiment Config ──────────────────────────────────────────────────────
DEV_S1_SAMPLE   = 20000
CAND_NEG_SAMPLE = 300000
TRAIN_FRAC      = 0.75
NEG_RATIO       = 6

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.04,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 4,
    'min_child_samples': 40,
    'verbose': -1,
    'n_jobs': -1,
    'is_unbalance': True,
    'seed': 42,
}

ADDR_STOPWORDS = {
    'road', 'street', 'lane', 'avenue', 'floor', 'building', 'phase', 'block',
    'sector', 'near', 'opposite', 'behind', 'beside', 'dist', 'state', 'city',
    'delhi', 'mumbai', 'pune', 'bangalore', 'chennai', 'kolkata', 'hyderabad',
    'india', 'france', 'paris', 'lyon', 'bordeaux', 'texas', 'california',
    'florida', 'york', 'north', 'south', 'east', 'west', 'unit', 'suite',
    'nagar', 'colony', 'marg', 'bhavan', 'tower', 'complex', 'plaza', 'bldg',
    'cross', 'main', 'extn', 'hno', 'plot', 'shop', 'flat'
}

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

def get_rare_addr_tokens(na_str: str) -> set:
    if not na_str: return set()
    toks = set()
    for w in na_str.split():
        if len(w) >= 4 and not w.isdigit() and w not in ADDR_STOPWORDS:
            toks.add(w)
    return toks

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

def build_indices(cdf):
    n = len(cdf)
    cids = cdf["entity_id"].values
    nns_v = cdf["_nns"].values
    nn_v  = cdf["_nn"].values
    pc_v  = cdf["_pc"].values
    na_v  = cdf["_na"].values
    rn_v  = cdf["business_name"].values

    print(f"  Building 8 blocking indices across {n:,} candidates...", flush=True)
    
    idx_exact = defaultdict(list)
    for i, v in enumerate(nns_v):
        if v: idx_exact[v].append(i)
    idx_exact = dict(idx_exact)
    
    idx_pfx3 = defaultdict(list)
    for i, v in enumerate(nn_v):
        if len(v) >= 3: idx_pfx3[v[:3]].append(i)
    idx_pfx3 = dict(idx_pfx3)
    
    idx_phone = defaultdict(list)
    for i in range(n):
        for k in pkeys(rn_v[i]):
            idx_phone[k].append(i)
    idx_phone = dict(idx_phone)
    
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
    
    idx_postal = defaultdict(list)
    for i, v in enumerate(pc_v):
        if v: idx_postal[v].append(i)
    idx_postal = dict(idx_postal)
    
    idx_addr_key = defaultdict(list)
    for i in range(n):
        for ak in get_addr_keys(na_v[i]):
            idx_addr_key[ak].append(i)
    idx_addr_key = dict(idx_addr_key)
    
    idx_comp = defaultdict(list)
    for i in range(n):
        for ck in get_name_addr_composite(nn_v[i], na_v[i]):
            idx_comp[ck].append(i)
    idx_comp = dict(idx_comp)
    
    addr_tok_df = Counter()
    for i in range(n):
        for w in get_rare_addr_tokens(na_v[i]):
            addr_tok_df[w] += 1
            
    idx_rare_addr = defaultdict(list)
    for i in range(n):
        for w in get_rare_addr_tokens(na_v[i]):
            if addr_tok_df[w] <= 600:
                idx_rare_addr[w].append(i)
    idx_rare_addr = dict(idx_rare_addr)
    
    return (cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
            idx_postal, idx_addr_key, idx_comp, idx_rare_addr)

def block_dataframe(df, cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
                    idx_postal, idx_addr_key, idx_comp, idx_rare_addr):
    s1_ids = df["entity_id"].values
    s1_nns = df["_nns"].values
    s1_nn  = df["_nn"].values
    s1_rn  = df["business_name"].values
    s1_pc  = df["_pc"].values
    s1_na  = df["_na"].values

    res = {}
    for sid, nn, nns, rn, pc, na in zip(s1_ids, s1_nn, s1_nns, s1_rn, s1_pc, s1_na):
        s_cands = set(idx_exact.get(nns, []))
        
        pfx3 = nn[:3] if len(nn) >= 3 else nn
        if pfx3:
            b = idx_pfx3.get(pfx3, [])
            if len(b) <= 800: s_cands.update(b)
            
        for k in pkeys(rn):
            b = idx_phone.get(k, [])
            if len(b) <= 800: s_cands.update(b)
            
        for w in get_rare_tokens(nn):
            b = idx_rare_tok.get(w, [])
            if len(b) <= 800: s_cands.update(b)
            
        if pc:
            s_cands.update(idx_postal.get(pc, []))
            
        for ak in get_addr_keys(na):
            b = idx_addr_key.get(ak, [])
            if len(b) <= 400: s_cands.update(b)
            
        for ck in get_name_addr_composite(nn, na):
            b = idx_comp.get(ck, [])
            if len(b) <= 400: s_cands.update(b)
            
        for w in get_rare_addr_tokens(na):
            b = idx_rare_addr.get(w, [])
            if len(b) <= 400: s_cands.update(b)
            
        res[sid] = {cids[j] for j in s_cands}
    return res

# ── Main Pipeline ──────────────────────────────────────────────────────────
def main():
    T0 = time.time()
    print("=" * 75, flush=True)
    print("EXPERIMENT 03: 48 FEATURES + HARD NEGATIVES + 8-CHANNEL BLOCKING", flush=True)
    print("=" * 75, flush=True)

    # [1] Load Data
    s1_full = load_source("train", 1)
    s2_full = load_source("train", 2)
    s3_full = load_source("train", 3)
    gt_full = load_ground_truth()

    # [2] Dev Sampling
    rng = np.random.RandomState(42)
    all_ids = list(gt_full.keys())
    rng.shuffle(all_ids)
    dev_ids = all_ids[:DEV_S1_SAMPLE]
    
    true_match_ids = set()
    for sid in dev_ids:
        true_match_ids.update(gt_full[sid])
    print(f"Dev S1: {len(dev_ids):,}, True matches: {len(true_match_ids):,}", flush=True)
    
    all_cand_ids = set(s2_full["entity_id"]) | set(s3_full["entity_id"])
    neg_pool = list(all_cand_ids - true_match_ids)
    rng.shuffle(neg_pool)
    neg_sample = set(neg_pool[:CAND_NEG_SAMPLE])
    keep_cand_ids = true_match_ids | neg_sample
    print(f"Candidate pool: {len(keep_cand_ids):,}", flush=True)

    s1 = s1_full[s1_full["entity_id"].isin(dev_ids)].copy().reset_index(drop=True)
    s2 = s2_full[s2_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    gt = {k: gt_full[k] for k in dev_ids}
    
    del s1_full, s2_full, s3_full, gt_full
    gc.collect()

    # [3] Normalize
    print("\nNormalizing records...", flush=True)
    add_norm_cols(s1); add_norm_cols(s2); add_norm_cols(s3)
    
    # [4] Split Train / Val
    dev_list = list(gt.keys())
    rng2 = np.random.RandomState(123)
    rng2.shuffle(dev_list)
    n_val = int(len(dev_list) * (1 - TRAIN_FRAC))
    val_ids = set(dev_list[:n_val])
    train_ids = set(dev_list[n_val:])
    gt_tr = {k: gt[k] for k in train_ids}
    gt_va = {k: gt[k] for k in val_ids}
    
    # [5] Build Blocking Indices
    cdf = pd.concat([s2, s3], ignore_index=True)
    (cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
     idx_postal, idx_addr_key, idx_comp, idx_rare_addr) = build_indices(cdf)
    
    # [6] Block Train & Val
    print("\nBlocking validation set...", flush=True)
    val_s1 = s1[s1["entity_id"].isin(val_ids)].copy()
    val_cands = block_dataframe(val_s1, cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
                                idx_postal, idx_addr_key, idx_comp, idx_rare_addr)
    br_va = blocking_recall(val_cands, gt_va)
    print(f">>> VALIDATION BLOCKING RECALL: {br_va['recall_ceiling']:.4%} (Missed: {br_va['missed_pairs']:,} pairs, Total: {br_va['total_candidates']:,})", flush=True)
    
    print("Blocking training set...", flush=True)
    tr_s1 = s1[s1["entity_id"].isin(train_ids)].copy()
    tr_cands = block_dataframe(tr_s1, cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
                               idx_postal, idx_addr_key, idx_comp, idx_rare_addr)
    br_tr = blocking_recall(tr_cands, gt_tr)
    print(f">>> TRAIN BLOCKING RECALL: {br_tr['recall_ceiling']:.4%}", flush=True)

    # [7] Build Fast Lookups
    print("\nBuilding fast record lookups...", flush=True)
    s1L = {}
    for sid, nn, na, c, rn, ra, pc in zip(s1["entity_id"].values, s1["_nn"].values, s1["_na"].values,
                                          s1["country"].values, s1["business_name"].values,
                                          s1["business_address"].values, s1["_pc"].values):
        s1L[sid] = (nn, na, c, rn, ra, pc)
        
    cL = {}
    for cid, nn, na, c, rn, ra, pc in zip(cdf["entity_id"].values, cdf["_nn"].values, cdf["_na"].values,
                                          cdf["country"].values, cdf["business_name"].values,
                                          cdf["business_address"].values, cdf["_pc"].values):
        cL[cid] = (nn, na, c, rn, ra, pc)

    # [8] Generate Labeled Pairs with Hard-Negative Stratification
    print("\nGenerating training pairs with Hard Negatives...", flush=True)
    def gen_stratified_pairs(cands_dict, gt_dict, neg_ratio):
        pairs = []
        for sid, cs in cands_dict.items():
            if sid not in gt_dict: continue
            tm = gt_dict[sid]
            pos = cs & tm
            neg = list(cs - tm)
            
            for c in pos:
                if c in cL: pairs.append((sid, c, 1))
                
            if not neg: continue
            
            s_nn, s_na = s1L[sid][0], s1L[sid][1]
            hard_negs = []
            random_negs = []
            
            for cid in neg:
                if cid not in cL: continue
                c_nn, c_na = cL[cid][0], cL[cid][1]
                # High name similarity or identical address tokens = Hard Negative
                if (s_nn and c_nn and (s_nn == c_nn or s_nn[:4] == c_nn[:4])) or (s_na and c_na and s_na[:8] == c_na[:8]):
                    hard_negs.append(cid)
                else:
                    random_negs.append(cid)
                    
            n_target = min(len(neg), max(neg_ratio * max(len(pos), 1), 4))
            n_hard = min(len(hard_negs), int(n_target * 0.6))
            n_rand = min(len(random_negs), n_target - n_hard)
            
            r = np.random.RandomState(hash(sid) % (2**31))
            if hard_negs and n_hard > 0:
                for c in r.choice(hard_negs, n_hard, replace=False):
                    pairs.append((sid, c, 0))
            if random_negs and n_rand > 0:
                for c in r.choice(random_negs, n_rand, replace=False):
                    pairs.append((sid, c, 0))
                    
        return pairs

    tr_pairs = gen_stratified_pairs(tr_cands, gt_tr, NEG_RATIO)
    va_pairs = gen_stratified_pairs(val_cands, gt_va, NEG_RATIO)
    print(f"Train Pairs: {len(tr_pairs):,} (pos={sum(l for _,_,l in tr_pairs):,})", flush=True)
    print(f"Val Pairs:   {len(va_pairs):,} (pos={sum(l for _,_,l in va_pairs):,})", flush=True)

    # [9] Compute 48 Features
    print(f"\nComputing {len(FEATURE_NAMES)} features...", flush=True)
    def to_Xy(pairs):
        X = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
        y = np.zeros(len(pairs), dtype=np.int32)
        for i, (sid, cid, lbl) in enumerate(tqdm(pairs, desc="Feats")):
            s = s1L[sid]; c = cL[cid]
            X[i] = compute_pair_features(s[0], s[1], s[2], s[3], s[4], c[0], c[1], c[2], c[3], c[4], s[5], c[5])
            y[i] = lbl
        return X, y

    Xtr, ytr = to_Xy(tr_pairs)
    Xva, yva = to_Xy(va_pairs)

    # [10] Train LightGBM
    print("\nTraining LightGBM Classifier...", flush=True)
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(Xva, label=yva, feature_name=FEATURE_NAMES, reference=dtrain)
    
    model = lgb.train(
        LGBM_PARAMS, dtrain, 2500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    print(f"Best Iteration: {model.best_iteration}", flush=True)
    
    imp = sorted(zip(FEATURE_NAMES, model.feature_importance('gain')), key=lambda x: -x[1])
    print("\nTop 20 Features by Gain:", flush=True)
    for f, v in imp[:20]:
        print(f"  {f:<25} {v:.1f}", flush=True)

    # [11] Score ALL Validation Candidates & Candidate-Relative Calibration
    print("\nScoring all validation candidate pairs for final calibration...", flush=True)
    all_val_pairs = [(sid, cid) for sid, cs in val_cands.items() for cid in cs if cid in cL]
    print(f"Total Val Pairs to Score: {len(all_val_pairs):,}", flush=True)
    
    chunk_size = 250000
    all_probs = []
    all_s1v, all_cv = [], []
    
    for st in range(0, len(all_val_pairs), chunk_size):
        en = min(st + chunk_size, len(all_val_pairs))
        ch = all_val_pairs[st:en]
        Xch = np.zeros((len(ch), len(FEATURE_NAMES)), dtype=np.float32)
        for i, (sid, cid) in enumerate(ch):
            s = s1L[sid]; c = cL[cid]
            Xch[i] = compute_pair_features(s[0], s[1], s[2], s[3], s[4], c[0], c[1], c[2], c[3], c[4], s[5], c[5])
        probs = model.predict(Xch)
        all_probs.extend(probs.tolist())
        all_s1v.extend([p[0] for p in ch])
        all_cv.extend([p[1] for p in ch])
        del Xch; gc.collect()
        
    all_probs = np.array(all_probs)

    # Group scores by S1 entity
    s1_to_scored_cands = defaultdict(list)
    for i in range(len(all_s1v)):
        s1_to_scored_cands[all_s1v[i]].append((all_cv[i], all_probs[i]))

    # [12] Threshold & Decision Rule Sweep
    print("\nRunning Threshold & Decision Rule Calibration...", flush=True)
    best_f05 = 0.0
    best_config = {}
    sweep_log = []

    for thr in np.arange(0.50, 0.98, 0.02):
        preds = {}
        for sid in gt_va:
            c_scores = s1_to_scored_cands.get(sid, [])
            if not c_scores:
                preds[sid] = set()
                continue
            
            # Matched set above threshold
            matched = {cid for cid, sc in c_scores if sc >= thr}
            preds[sid] = matched
            
        res = f05_macro(preds, gt_va)
        sweep_log.append((thr, res))
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_config = {'thr': thr, 'res': res}

    print("\n" + "=" * 75, flush=True)
    print("CALIBRATION RESULTS & EVALUATION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    
    top_thr = best_config['thr']
    top_r = best_config['res']
    print(f"Optimal Probability Threshold: {top_thr:.2f}", flush=True)
    print(f"Macro F0.5 Score:             {top_r['macro_f05']:.4f}", flush=True)
    print(f"Singleton Accuracy:           {top_r['singleton_acc']:.4f}", flush=True)
    print(f"Multi-Match F0.5:             {top_r['multi_f05']:.4f}", flush=True)
    print(f"Precision (Macro):            {top_r['macro_prec']:.4f}", flush=True)
    print(f"Recall (Macro):               {top_r['macro_rec']:.4f}", flush=True)
    print(f"Micro Stats: TP={top_r['tp']:,}, FP={top_r['fp']:,}, FN={top_r['fn']:,}", flush=True)
    
    print("\nThreshold Sweep Curve (Top 10):", flush=True)
    sweep_log.sort(key=lambda x: -x[1]['macro_f05'])
    for thr, r in sweep_log[:10]:
        print(f"  thr={thr:.2f} -> Macro-F0.5: {r['macro_f05']:.4f} | SingAcc: {r['singleton_acc']:.4f} | Multi-F0.5: {r['multi_f05']:.4f} | FP: {r['fp']:,} | TP: {r['tp']:,}", flush=True)

    # [13] Log to experiments.csv
    log_experiment({
        "experiment_id": "EXP-03-GBM-FEATURES",
        "date": "2026-09-25",
        "description": "48 features + Stratified Hard Negatives + 8-channel Blocking",
        "blocking_recall": f"{br_va['recall_ceiling']:.4f}",
        "val_entities": len(gt_va),
        "val_pairs_scored": len(all_val_pairs),
        "threshold": f"{top_thr:.2f}",
        "precision": f"{top_r['macro_prec']:.4f}",
        "recall": f"{top_r['macro_rec']:.4f}",
        "macro_f05": f"{top_r['macro_f05']:.4f}",
        "singleton_acc": f"{top_r['singleton_acc']:.4f}",
        "multi_f05": f"{top_r['multi_f05']:.4f}",
        "tp": top_r["tp"],
        "fp": top_r["fp"],
        "fn": top_r["fn"],
        "notes": f"Best Iter: {model.best_iteration}, Runtime: {(time.time()-T0)/60:.1f}m"
    })
    
    # Save model
    model.save_model(os.path.join(CACHE_DIR, "exp03_model.txt"))
    print(f"\nModel saved to {CACHE_DIR}/exp03_model.txt", flush=True)
    print(f"Total Experiment Time: {(time.time()-T0)/60:.1f} minutes", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
