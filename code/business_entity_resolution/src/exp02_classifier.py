#!/usr/bin/env python3
"""
Experiment 02: Baseline classifier + threshold calibration.

Depends on: exp01 blocking results (or re-runs blocking).
Goal: Train LightGBM on pairwise features, sweep threshold for F0.5.

Steps:
  1. Load data, normalize
  2. Split train/val (entity-level)
  3. Block train + val sets
  4. Generate labeled pairs (pos + sampled neg)
  5. Compute features
  6. Train LightGBM
  7. Score ALL val candidates
  8. Sweep threshold → maximize macro F0.5
  9. Report: F0.5, singleton acc, multi-match F0.5
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
from metaphone import doublemetaphone
from tqdm import tqdm
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_source, load_ground_truth, add_norm_cols,
                  norm_name, norm_name_sorted, norm_addr, get_postal, get_nums,
                  blocking_recall, entity_split, print_split_stats,
                  f05_macro, CACHE_DIR)

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_SAMPLE = 50000    # S1 entities for training
VAL_SAMPLE   = 15000    # S1 entities for validation
NEG_RATIO    = 5
TFIDF_TOP_K  = 10

LGBM_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'boosting_type': 'gbdt', 'num_leaves': 127,
    'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'verbose': -1, 'n_jobs': -1,
    'is_unbalance': True, 'seed': 42,
}

# ── Blocking (reuse from exp01) ───────────────────────────────────────────

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

def build_blocking_indices(cands_df):
    """Build hash-based blocking indices."""
    n = len(cands_df)
    cand_ids = cands_df["entity_id"].values
    nns_v = cands_df["_nns"].values
    nn_v  = cands_df["_nn"].values
    pc_v  = cands_df["_pc"].values
    na_v  = cands_df["_na"].values
    raw_n = cands_df["business_name"].values
    
    print("  Building indices...")
    
    idx_name = defaultdict(list)
    for i, v in enumerate(nns_v):
        if v: idx_name[v].append(i)
    
    idx_pfx = defaultdict(list)
    for i, v in enumerate(nn_v):
        if len(v) >= 4: idx_pfx[v[:4]].append(i)
    
    idx_phone = defaultdict(list)
    for i in tqdm(range(n), desc="  Phonetic idx", mininterval=15):
        for k in phonetic_keys(raw_n[i]):
            idx_phone[k].append(i)
    
    idx_postal = defaultdict(list)
    for i, v in enumerate(pc_v):
        if v: idx_postal[v].append(i)
    
    idx_nums = defaultdict(list)
    for i in range(n):
        for num in get_nums(na_v[i]):
            if len(num) >= 2: idx_nums[num].append(i)
    
    print(f"    name={len(idx_name):,}, pfx={len(idx_pfx):,}, "
          f"phone={len(idx_phone):,}, postal={len(idx_postal):,}, nums={len(idx_nums):,}")
    
    return cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums


def block_entity(row, cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums):
    """Get candidate IDs for one S1 entity (union of strategies)."""
    cands = set()
    
    # A: exact name
    nns_val = row["_nns"]
    if nns_val and nns_val in idx_name:
        cands.update(idx_name[nns_val])
    
    # B: prefix
    nn_val = row["_nn"]
    pfx = nn_val[:4] if len(nn_val) >= 4 else nn_val
    if pfx and pfx in idx_pfx:
        bucket = idx_pfx[pfx]
        if len(bucket) < 50000: cands.update(bucket)
    
    # C: phonetic
    for k in phonetic_keys(row["business_name"]):
        bucket = idx_phone.get(k)
        if bucket and len(bucket) < 10000:
            cands.update(bucket)
    
    # D: postal
    pc = row["_pc"]
    if pc and pc in idx_postal:
        cands.update(idx_postal[pc])
    
    # E: address numbers (≥2 shared)
    nums_list = get_nums(row["_na"])
    if len(nums_list) >= 2:
        hits = defaultdict(int)
        for num in nums_list:
            if len(num) >= 2 and num in idx_nums:
                for idx in idx_nums[num]:
                    hits[idx] += 1
        for idx, cnt in hits.items():
            if cnt >= 2: cands.add(idx)
    
    return {cand_ids[j] for j in cands}


def block_all(s1_df, cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums):
    """Block all S1 entities."""
    result = {}
    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df),
                       desc="  Blocking", mininterval=15):
        result[row["entity_id"]] = block_entity(
            row, cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums)
    return result


# ── Features ──────────────────────────────────────────────────────────────

FEAT_NAMES = [
    'name_jaccard', 'name_tri_jacc', 'name_lev', 'name_tsort', 'name_tset',
    'name_partial', 'name_wr', 'name_exact', 'name_exact_sorted',
    'name_len_d', 'name_len_r', 'name_tok_d',
    'addr_jaccard', 'addr_tri_jacc', 'addr_lev', 'addr_tsort', 'addr_tset',
    'addr_partial', 'addr_len_d', 'addr_len_r', 'addr_tok_d',
    'post_match', 'post_avail', 'num_jacc',
    'country_m', 'comb_tsort', 'comb_tset',
    'name_empty', 'addr_empty', 'phone_first_m',
]

def _tri(s):
    return set(s[i:i+3] for i in range(max(0, len(s)-2)))

def compute_features(s1nn, s1na, s1c, s1rn, s1ra,
                     cnn, cna, cc, crn, cra):
    """30 features for one (S1, candidate) pair."""
    # Name
    s1t = set(s1nn.split()) if s1nn else set()
    ct  = set(cnn.split()) if cnn else set()
    nj = len(s1t&ct)/max(len(s1t|ct),1) if (s1t or ct) else 0.
    sg, cg = _tri(s1nn), _tri(cnn)
    ntj = len(sg&cg)/max(len(sg|cg),1) if (sg or cg) else 0.
    nl  = fuzz.ratio(s1nn, cnn)/100.
    nts = fuzz.token_sort_ratio(s1nn, cnn)/100.
    ntse = fuzz.token_set_ratio(s1nn, cnn)/100.
    np_ = fuzz.partial_ratio(s1nn, cnn)/100.
    nw  = fuzz.WRatio(s1nn, cnn)/100.
    ne  = 1. if (s1nn and s1nn==cnn) else 0.
    ss = ' '.join(sorted(s1nn.split())) if s1nn else ''
    cs = ' '.join(sorted(cnn.split())) if cnn else ''
    nes = 1. if (ss and ss==cs) else 0.
    nld = abs(len(s1nn)-len(cnn))
    nlr = min(len(s1nn),len(cnn))/max(len(s1nn),len(cnn),1)
    ntd = abs(len(s1t)-len(ct))
    # Address
    s1at = set(s1na.split()) if s1na else set()
    cat  = set(cna.split()) if cna else set()
    aj = len(s1at&cat)/max(len(s1at|cat),1) if (s1at or cat) else 0.
    sag, cag = _tri(s1na), _tri(cna)
    atj = len(sag&cag)/max(len(sag|cag),1) if (sag or cag) else 0.
    al  = fuzz.ratio(s1na, cna)/100.
    ats = fuzz.token_sort_ratio(s1na, cna)/100.
    atse = fuzz.token_set_ratio(s1na, cna)/100.
    ap  = fuzz.partial_ratio(s1na, cna)/100.
    ald = abs(len(s1na)-len(cna))
    alr = min(len(s1na),len(cna))/max(len(s1na),len(cna),1)
    atd = abs(len(s1at)-len(cat))
    # Postal/numbers
    s1p, cp_ = get_postal(s1ra), get_postal(cra)
    pm = 1. if (s1p and cp_ and s1p==cp_) else 0.
    pa = 1. if (s1p and cp_) else 0.
    sn, cn = set(get_nums(s1ra)), set(get_nums(cra))
    numj = len(sn&cn)/max(len(sn|cn),1) if (sn or cn) else 0.
    # Cross
    cm = 1. if s1c==cc else 0.
    cts_ = (nts+ats)/2.
    ctse_ = (ntse+atse)/2.
    nme = 1. if not s1nn else 0.
    ade = 1. if not s1na else 0.
    # Phonetic first token
    phm = 0.
    try:
        f1 = s1nn.split()[0] if s1nn else ''
        f2 = cnn.split()[0] if cnn else ''
        if f1 and f2:
            p1,_ = doublemetaphone(f1); p2,_ = doublemetaphone(f2)
            if p1 and p2 and p1==p2: phm=1.
    except: pass

    return np.array([nj,ntj,nl,nts,ntse,np_,nw,ne,nes,nld,nlr,ntd,
                     aj,atj,al,ats,atse,ap,ald,alr,atd,
                     pm,pa,numj,cm,cts_,ctse_,nme,ade,phm], dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    T0 = time.time()
    print("=" * 70)
    print("EXPERIMENT 02: BASELINE CLASSIFIER + THRESHOLD")
    print("=" * 70)

    # [1] Load
    print("\n[1] Load data")
    s1 = load_source("train", 1)
    s2 = load_source("train", 2)
    s3 = load_source("train", 3)
    gt = load_ground_truth()
    
    # [2] Normalize
    print("\n[2] Normalize")
    add_norm_cols(s1); add_norm_cols(s2); add_norm_cols(s3)
    
    # [3] Split
    print(f"\n[3] Entity-level split (train={TRAIN_SAMPLE:,}, val={VAL_SAMPLE:,})")
    rng = np.random.RandomState(42)
    all_ids = list(gt.keys()); rng.shuffle(all_ids)
    sings = [x for x in all_ids if not gt[x]]
    nons  = [x for x in all_ids if gt[x]]
    sf = len(sings)/len(all_ids)
    
    vs, vn = int(VAL_SAMPLE*sf), VAL_SAMPLE - int(VAL_SAMPLE*sf)
    val_ids = set(sings[:vs] + nons[:vn])
    ts, tn = int(TRAIN_SAMPLE*sf), TRAIN_SAMPLE - int(TRAIN_SAMPLE*sf)
    rem_s = [x for x in sings if x not in val_ids]
    rem_n = [x for x in nons if x not in val_ids]
    train_ids = set(rem_s[:ts] + rem_n[:tn])
    
    gt_tr = {k:gt[k] for k in train_ids}
    gt_va = {k:gt[k] for k in val_ids}
    print_split_stats("Train", gt_tr)
    print_split_stats("Val",   gt_va)
    
    # [4] Blocking
    print("\n[4] Build blocking indices")
    cdf = pd.concat([s2, s3], ignore_index=True)
    cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums = \
        build_blocking_indices(cdf)
    
    print("\n  Block validation set...")
    val_s1 = s1[s1["entity_id"].isin(val_ids)].copy()
    val_cands = block_all(val_s1, cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums)
    br = blocking_recall(val_cands, gt_va)
    print(f"  Val blocking recall: {br['recall_ceiling']:.4f}")
    
    print("  Block training set...")
    tr_s1 = s1[s1["entity_id"].isin(train_ids)].copy()
    tr_cands = block_all(tr_s1, cand_ids, idx_name, idx_pfx, idx_phone, idx_postal, idx_nums)
    br_tr = blocking_recall(tr_cands, gt_tr)
    print(f"  Train blocking recall: {br_tr['recall_ceiling']:.4f}")
    
    # [5] Lookups
    print("\n[5] Build lookups")
    needed_s1 = train_ids | val_ids
    s1L = {}
    for _, r in s1.iterrows():
        if r["entity_id"] in needed_s1:
            s1L[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                    r["business_name"],r["business_address"])
    needed_c = set()
    for cs_ in tr_cands.values(): needed_c.update(cs_)
    for cs_ in val_cands.values(): needed_c.update(cs_)
    cL = {}
    for _, r in cdf.iterrows():
        if r["entity_id"] in needed_c:
            cL[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                   r["business_name"],r["business_address"])
    print(f"  S1L: {len(s1L):,}, cL: {len(cL):,}")
    
    # [6] Generate pairs
    print("\n[6] Generate labeled pairs")
    def gen_pairs(cd, gd, nr):
        pairs = []
        for sid, cs_ in cd.items():
            if sid not in gd: continue
            tm = gd[sid]; pos = cs_ & tm; neg = list(cs_-tm)
            for c in pos:
                if c in cL: pairs.append((sid,c,1))
            nn_ = min(len(neg), max(nr*max(len(pos),1), 3))
            if neg:
                r = np.random.RandomState(hash(sid)%(2**31))
                for c in r.choice(neg, min(nn_,len(neg)), replace=False):
                    if c in cL: pairs.append((sid,c,0))
        return pairs
    
    tr_pairs = gen_pairs(tr_cands, gt_tr, NEG_RATIO)
    va_pairs = gen_pairs(val_cands, gt_va, NEG_RATIO)
    print(f"  Train: {len(tr_pairs):,} (pos={sum(l for _,_,l in tr_pairs):,})")
    print(f"  Val:   {len(va_pairs):,} (pos={sum(l for _,_,l in va_pairs):,})")
    
    # [7] Features
    print("\n[7] Compute features")
    def to_Xy(pairs):
        X = np.zeros((len(pairs), len(FEAT_NAMES)), dtype=np.float32)
        y = np.zeros(len(pairs), dtype=np.int32)
        s1s, css = [], []
        for i,(sid,cid,lbl) in enumerate(tqdm(pairs, desc="  Feats", mininterval=15)):
            s = s1L[sid]; c = cL[cid]
            X[i] = compute_features(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
            y[i] = lbl; s1s.append(sid); css.append(cid)
        return X, y, s1s, css
    
    Xtr, ytr, _, _ = to_Xy(tr_pairs)
    print(f"  Xtr: {Xtr.shape}")
    Xva, yva, va_s1s, va_cs = to_Xy(va_pairs)
    print(f"  Xva: {Xva.shape}")
    
    # [8] Train LightGBM
    print("\n[8] Train LightGBM")
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=FEAT_NAMES)
    dval   = lgb.Dataset(Xva, label=yva, feature_name=FEAT_NAMES, reference=dtrain)
    mdl = lgb.train(LGBM_PARAMS, dtrain, 1500, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"  Best iter: {mdl.best_iteration}")
    imp = sorted(zip(FEAT_NAMES, mdl.feature_importance('gain')), key=lambda x:-x[1])
    print("\n  Top features:")
    for f,v in imp[:15]: print(f"    {f}: {v:.0f}")
    
    # [9] Threshold sweep on ALL val candidates
    print("\n[9] Threshold calibration")
    print("  Scoring all val candidates...")
    avp = [(sid, cid) for sid, cs_ in val_cands.items()
           for cid in cs_ if cid in cL]
    print(f"    {len(avp):,} pairs to score")
    
    chunk = 200000
    all_pr, all_s1v, all_cv = [], [], []
    for st in range(0, len(avp), chunk):
        en = min(st+chunk, len(avp))
        ch = avp[st:en]
        Xch = np.zeros((len(ch), len(FEAT_NAMES)), dtype=np.float32)
        for i,(sid,cid) in enumerate(tqdm(ch, desc=f"  Chunk{st//chunk+1}", mininterval=15)):
            s = s1L[sid]; c = cL[cid]
            Xch[i] = compute_features(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
        pr = mdl.predict(Xch)
        all_pr.extend(pr.tolist())
        all_s1v.extend([p[0] for p in ch])
        all_cv.extend([p[1] for p in ch])
        del Xch; gc.collect()
    
    all_pr = np.array(all_pr)
    
    print("\n  Sweeping thresholds...")
    best_f, best_t = 0., 0.5
    sweep = []
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = {s: set() for s in gt_va}
        for j in np.where(all_pr >= thr)[0]:
            s = all_s1v[j]
            if s in preds: preds[s].add(all_cv[j])
        r = f05_macro(preds, gt_va)
        sweep.append((thr, r))
        if r['f05'] > best_f: best_f = r['f05']; best_t = thr
    
    # Report
    r = [x[1] for x in sweep if abs(x[0]-best_t)<0.001][0]
    print(f"\n  *** BEST THRESHOLD: {best_t:.2f} ***")
    print(f"      F_0.5 macro:     {r['f05']:.4f}")
    print(f"      Singleton acc:   {r['sing_acc']:.4f}")
    print(f"      Multi-match F05: {r['multi_f05']:.4f}")
    print(f"      TP={r['tp']:,}, FP={r['fp']:,}, FN={r['fn']:,}")
    print(f"      Pred empty={r['pred_empty']:,}, nonempty={r['pred_nonempty']:,}")
    
    # Top-10 thresholds
    sweep.sort(key=lambda x: -x[1]['f05'])
    print("\n  Top-10 thresholds:")
    for thr, r in sweep[:10]:
        print(f"    thr={thr:.2f}: F05={r['f05']:.4f} sing={r['sing_acc']:.4f} multi={r['multi_f05']:.4f}")
    
    # Save model
    mdl.save_model(os.path.join(CACHE_DIR, "exp02_model.txt"))
    print(f"\n  Total time: {(time.time()-T0)/60:.1f} min")
    print("  DONE.")


if __name__ == "__main__":
    main()
