#!/usr/bin/env python3
"""
Find exact optimal decision thresholds for Macro F0.5 on training validation set.
"""
import os, sys, time, gc
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_source, load_ground_truth, add_norm_cols, f05_macro
from features import FEATURE_NAMES, compute_pair_features
from exp03_pipeline import build_indices, block_dataframe

def main():
    print("Loading model and data for threshold sweep...")
    model = lgb.Booster(model_file="models/final_model.txt")
    
    s1_full = load_source("train", 1)
    s2_full = load_source("train", 2)
    s3_full = load_source("train", 3)
    gt_full = load_ground_truth()

    # Sample 3,000 S1 entities
    rng = np.random.RandomState(42)
    all_ids = list(gt_full.keys())
    rng.shuffle(all_ids)
    sample_s1_ids = set(all_ids[:3000])
    
    gt_sample = {k: gt_full[k] for k in sample_s1_ids}
    true_match_ids = set()
    for sid in sample_s1_ids:
        true_match_ids.update(gt_full[sid])
        
    all_cand_ids = set(s2_full["entity_id"]) | set(s3_full["entity_id"])
    neg_pool = list(all_cand_ids - true_match_ids)
    rng.shuffle(neg_pool)
    neg_sample = set(neg_pool[:500000]) # 500k negative pool
    keep_cand_ids = true_match_ids | neg_sample

    s1 = s1_full[s1_full["entity_id"].isin(sample_s1_ids)].copy().reset_index(drop=True)
    s2 = s2_full[s2_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    
    del s1_full, s2_full, s3_full, gt_full
    gc.collect()

    add_norm_cols(s1); add_norm_cols(s2); add_norm_cols(s3)
    
    cdf = pd.concat([s2, s3], ignore_index=True)
    (cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
     idx_postal, idx_addr_key, idx_comp, idx_rare_addr) = build_indices(cdf)
    
    print("Blocking 3,000 entities against 500k candidate pool...")
    val_cands = block_dataframe(s1, cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
                                idx_postal, idx_addr_key, idx_comp, idx_rare_addr)

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

    print(f"Extracting features for candidate pairs...")
    X_list = []
    pair_index = []
    
    for sid, c_set in val_cands.items():
        s_info = s1L[sid]
        for cid in c_set:
            if cid in cL:
                c_info = cL[cid]
                feats = compute_pair_features(*s_info, *c_info)
                X_list.append(feats)
                pair_index.append((sid, cid))

    X_mat = np.array(X_list, dtype=np.float32)
    print(f"Total candidate pairs: {len(X_mat):,}. Predicting model probabilities...")
    probs = model.predict(X_mat)

    # Score across multiple thresholds
    print("\n" + "=" * 70)
    print(f"{'Threshold':>10} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Singletons':>12} | {'Total Preds':>12}")
    print("=" * 70)

    best_thr = 0.5
    best_f05 = 0.0

    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97]:
        preds = defaultdict(set)
        for idx, p in enumerate(probs):
            if p >= thr:
                sid, cid = pair_index[idx]
                preds[sid].add(cid)
                
        # Fill empty for singletons
        pred_dict = {sid: preds[sid] for sid in sample_s1_ids}
        
        # Compute Macro F0.5
        metric_res = f05_macro(pred_dict, gt_sample)
        f05 = metric_res["macro_f05"]
        p_val = metric_res["macro_prec"]
        r_val = metric_res["macro_rec"]
        n_preds = sum(len(v) for v in pred_dict.values())
        n_sing = sum(1 for v in pred_dict.values() if len(v) == 0)

        if f05 > best_f05:
            best_f05 = f05
            best_thr = thr

        print(f"{thr:10.2f} | {f05:10.4f} | {p_val:10.4f} | {r_val:10.4f} | {n_sing:6d} ({n_sing/len(sample_s1_ids):.1%}) | {n_preds:12,d}")

    print("=" * 70)
    print(f"Optimal Threshold: {best_thr:.2f} with Macro F0.5: {best_f05:.4f}")

if __name__ == "__main__":
    main()
