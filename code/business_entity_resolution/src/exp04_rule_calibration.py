#!/usr/bin/env python3
"""
Experiment 04: Tiered Decision Rules & Evidence Calibration.
Loads the trained exp03 LightGBM model, evaluates validation set with:
  1. Tiered Evidence Thresholds (Missing Address Protection, High Concordance Boost)
  2. Contradiction Veto Filter (Postal mismatch + addr mismatch)
  3. Candidate-Relative Margin Pruning
  4. Measure Macro F0.5, Singleton Accuracy, Multi-Match F0.5
"""
import os, sys, time, gc
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_source, load_ground_truth, add_norm_cols,
                  f05_macro, log_experiment, CACHE_DIR)
from features import FEATURE_NAMES, compute_pair_features
from exp03_pipeline import (DEV_S1_SAMPLE, CAND_NEG_SAMPLE, TRAIN_FRAC,
                            build_indices, block_dataframe)

def main():
    T0 = time.time()
    print("=" * 75, flush=True)
    print("EXPERIMENT 04: TIERED DECISION RULES & EVIDENCE CALIBRATION", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Data & Ground Truth
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
        
    all_cand_ids = set(s2_full["entity_id"]) | set(s3_full["entity_id"])
    neg_pool = list(all_cand_ids - true_match_ids)
    rng.shuffle(neg_pool)
    neg_sample = set(neg_pool[:CAND_NEG_SAMPLE])
    keep_cand_ids = true_match_ids | neg_sample

    s1 = s1_full[s1_full["entity_id"].isin(dev_ids)].copy().reset_index(drop=True)
    s2 = s2_full[s2_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    gt = {k: gt_full[k] for k in dev_ids}
    
    del s1_full, s2_full, s3_full, gt_full
    gc.collect()

    add_norm_cols(s1); add_norm_cols(s2); add_norm_cols(s3)
    
    dev_list = list(gt.keys())
    rng2 = np.random.RandomState(123)
    rng2.shuffle(dev_list)
    n_val = int(len(dev_list) * (1 - TRAIN_FRAC))
    val_ids = set(dev_list[:n_val])
    gt_va = {k: gt[k] for k in val_ids}
    
    cdf = pd.concat([s2, s3], ignore_index=True)
    (cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
     idx_postal, idx_addr_key, idx_comp, idx_rare_addr) = build_indices(cdf)
    
    print("\nBlocking validation set...", flush=True)
    val_s1 = s1[s1["entity_id"].isin(val_ids)].copy()
    val_cands = block_dataframe(val_s1, cids, idx_exact, idx_pfx3, idx_phone, idx_rare_tok,
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

    # 2. Load Trained Model
    model_path = os.path.join(CACHE_DIR, "exp03_model.txt")
    print(f"\nLoading trained model from {model_path}...", flush=True)
    model = lgb.Booster(model_file=model_path)

    # 3. Score Val Pairs with feature caching
    all_val_pairs = [(sid, cid) for sid, cs in val_cands.items() for cid in cs if cid in cL]
    print(f"Scoring {len(all_val_pairs):,} validation pairs...", flush=True)
    
    chunk_size = 300000
    all_probs = []
    all_s1v, all_cv = [], []
    all_flags = [] # Store (both_addr, cand_addr_missing, name_tsort, addr_tset, contradiction)
    
    for st in range(0, len(all_val_pairs), chunk_size):
        en = min(st + chunk_size, len(all_val_pairs))
        ch = all_val_pairs[st:en]
        Xch = np.zeros((len(ch), len(FEATURE_NAMES)), dtype=np.float32)
        flags_ch = []
        for i, (sid, cid) in enumerate(ch):
            s = s1L[sid]; c = cL[cid]
            feat = compute_pair_features(s[0], s[1], s[2], s[3], s[4], c[0], c[1], c[2], c[3], c[4], s[5], c[5])
            Xch[i] = feat
            # Store key flags: both_addr_present, cand_addr_missing, name_tsort, addr_tset, contradiction
            flags_ch.append((feat[29], feat[31], feat[3], feat[15], feat[41]))
            
        probs = model.predict(Xch)
        all_probs.extend(probs.tolist())
        all_s1v.extend([p[0] for p in ch])
        all_cv.extend([p[1] for p in ch])
        all_flags.extend(flags_ch)
        del Xch; gc.collect()

    print("\nEvaluating Tiered Decision Rules on Validation Set...", flush=True)
    
    # Grid search over tiered parameters
    best_res = None
    best_params = None
    best_f05 = 0.0

    for base_thr in [0.92, 0.94, 0.95, 0.96, 0.97]:
        for missing_addr_thr in [0.97, 0.98, 0.99]:
            for high_conf_thr in [0.85, 0.88, 0.90]:
                for margin in [0.10, 0.15, 0.20, 1.0]: # 1.0 = no margin pruning
                    preds = defaultdict(set)
                    
                    # Group by S1
                    s1_scored = defaultdict(list)
                    for i in range(len(all_s1v)):
                        sid = all_s1v[i]
                        cid = all_cv[i]
                        p = all_probs[i]
                        both_a, cand_a_miss, n_tsort, a_tset, contra = all_flags[i]
                        
                        # Rule Logic:
                        # 1. Contradiction Veto:
                        if contra >= 0.8 and p < 0.995:
                            continue
                            
                        # 2. Tiered Thresholds:
                        if both_a == 1.0 and n_tsort >= 0.80 and a_tset >= 0.80:
                            effective_thr = high_conf_thr
                        elif cand_a_miss == 1.0 or both_a == 0.0:
                            effective_thr = missing_addr_thr
                        else:
                            effective_thr = base_thr
                            
                        if p >= effective_thr:
                            s1_scored[sid].append((cid, p))
                            
                    # Apply Margin Pruning per S1
                    for sid, cand_list in s1_scored.items():
                        if not cand_list: continue
                        cand_list_sorted = sorted(cand_list, key=lambda x: -x[1])
                        top_p = cand_list_sorted[0][1]
                        for cid, p in cand_list_sorted:
                            if (top_p - p) <= margin:
                                preds[sid].add(cid)
                                
                    res = f05_macro(preds, gt_va)
                    if res['macro_f05'] > best_f05:
                        best_f05 = res['macro_f05']
                        best_res = res
                        best_params = {
                            'base_thr': base_thr,
                            'missing_addr_thr': missing_addr_thr,
                            'high_conf_thr': high_conf_thr,
                            'margin': margin
                        }

    print("\n" + "=" * 75, flush=True)
    print("OPTIMAL TIERED DECISION RULE RESULTS", flush=True)
    print("=" * 75, flush=True)
    print(f"Optimal Parameters: {best_params}", flush=True)
    print(f"Macro F0.5 Score:   {best_res['macro_f05']:.4f}", flush=True)
    print(f"Singleton Accuracy: {best_res['singleton_acc']:.4f}", flush=True)
    print(f"Multi-Match F0.5:   {best_res['multi_f05']:.4f}", flush=True)
    print(f"Macro Precision:    {best_res['macro_prec']:.4f}", flush=True)
    print(f"Macro Recall:       {best_res['macro_rec']:.4f}", flush=True)
    print(f"Micro Stats: TP={best_res['tp']:,}, FP={best_res['fp']:,}, FN={best_res['fn']:,}", flush=True)
    print("=" * 75, flush=True)

    log_experiment({
        "experiment_id": "EXP-04-TIERED-RULES",
        "date": "2026-09-25",
        "description": "Tiered Evidence Thresholding + Contradiction Veto + Margin Pruning",
        "blocking_recall": "0.9869",
        "val_entities": len(gt_va),
        "val_pairs_scored": len(all_val_pairs),
        "threshold": f"Base:{best_params['base_thr']}_Miss:{best_params['missing_addr_thr']}",
        "precision": f"{best_res['macro_prec']:.4f}",
        "recall": f"{best_res['macro_rec']:.4f}",
        "macro_f05": f"{best_res['macro_f05']:.4f}",
        "singleton_acc": f"{best_res['singleton_acc']:.4f}",
        "multi_f05": f"{best_res['multi_f05']:.4f}",
        "tp": best_res["tp"],
        "fp": best_res["fp"],
        "fn": best_res["fn"],
        "notes": f"Params: {best_params}, Runtime: {(time.time()-T0)/60:.1f}m"
    })

if __name__ == "__main__":
    main()
