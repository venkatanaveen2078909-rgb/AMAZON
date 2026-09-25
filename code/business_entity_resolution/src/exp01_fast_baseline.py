#!/usr/bin/env python3
"""
Experiment 01 (fast): Blocking recall + baseline classifier on dev sample.

Strategy for fast iteration at scale:
  - Sample 20K S1 entities for dev
  - Identify which S2/S3 IDs are true matches for sampled S1
  - Subsample S2/S3 to: true matches + 200K random negatives
  - Build blocking indices on this manageable candidate pool
  - Measure blocking recall
  - Train classifier + sweep threshold

This gives fast feedback (minutes not hours) while still being representative.
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
                  norm_name, norm_addr, get_postal, get_nums,
                  blocking_recall, entity_split, print_split_stats,
                  f05_macro, CACHE_DIR)

# ── Config ─────────────────────────────────────────────────────────────────
DEV_S1_SAMPLE = 20000          # S1 entities for dev
CAND_NEG_SAMPLE = 300000       # random S2/S3 negatives to include
TRAIN_FRAC = 0.75              # of dev sample
NEG_RATIO = 5
FLUSH = True                   # Force print flushing

def pr(msg): print(msg, flush=FLUSH)

# ── Phonetics ──────────────────────────────────────────────────────────────
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

# ── Blocking ───────────────────────────────────────────────────────────────
def build_indices(cdf):
    """Build blocking indices on candidate df."""
    n = len(cdf)
    cids = cdf["entity_id"].values
    nns = cdf["_nns"].values
    nn_v = cdf["_nn"].values
    pc = cdf["_pc"].values
    na = cdf["_na"].values
    rn = cdf["business_name"].values

    idx = {}
    
    # A: exact sorted name
    t0 = time.time()
    d = defaultdict(list)
    for i, v in enumerate(nns):
        if v: d[v].append(i)
    idx['name'] = dict(d)
    pr(f"    [A] Name exact:  {len(idx['name']):,} keys ({time.time()-t0:.1f}s)")

    # B: prefix-4
    t0 = time.time()
    d = defaultdict(list)
    for i, v in enumerate(nn_v):
        if len(v) >= 4: d[v[:4]].append(i)
    idx['pfx'] = dict(d)
    pr(f"    [B] Prefix-4:    {len(idx['pfx']):,} keys ({time.time()-t0:.1f}s)")

    # C: phonetic
    t0 = time.time()
    d = defaultdict(list)
    for i in range(n):
        for k in phonetic_keys(rn[i]):
            d[k].append(i)
    idx['phone'] = dict(d)
    pr(f"    [C] Phonetic:    {len(idx['phone']):,} keys ({time.time()-t0:.1f}s)")

    # D: postal
    t0 = time.time()
    d = defaultdict(list)
    for i, v in enumerate(pc):
        if v: d[v].append(i)
    idx['postal'] = dict(d)
    pr(f"    [D] Postal:      {len(idx['postal']):,} keys ({time.time()-t0:.1f}s)")

    # E: address numbers
    t0 = time.time()
    d = defaultdict(list)
    for i in range(n):
        for num in get_nums(na[i]):
            if len(num) >= 2: d[num].append(i)
    idx['nums'] = dict(d)
    pr(f"    [E] Addr nums:   {len(idx['nums']):,} keys ({time.time()-t0:.1f}s)")

    return cids, idx


def block_one(row, cids, idx):
    """Union of all strategies for one S1 entity."""
    results_per = {}
    
    # A: exact name
    nns = row["_nns"]
    a = set()
    if nns and nns in idx['name']:
        a = set(idx['name'][nns])
    results_per['A'] = a
    
    # B: prefix
    nn_v = row["_nn"]
    b = set()
    pfx = nn_v[:4] if len(nn_v) >= 4 else nn_v
    if pfx and pfx in idx['pfx']:
        bucket = idx['pfx'][pfx]
        if len(bucket) < 50000: b = set(bucket)
    results_per['B'] = b
    
    # C: phonetic
    c = set()
    for k in phonetic_keys(row["business_name"]):
        bucket = idx['phone'].get(k)
        if bucket and len(bucket) < 10000:
            c.update(bucket)
    results_per['C'] = c
    
    # D: postal
    d = set()
    pc = row["_pc"]
    if pc and pc in idx['postal']:
        d = set(idx['postal'][pc])
    results_per['D'] = d
    
    # E: addr nums (≥2 shared)
    e = set()
    nums = get_nums(row["_na"])
    if len(nums) >= 2:
        hits = defaultdict(int)
        for num in nums:
            if len(num) >= 2 and num in idx['nums']:
                for i in idx['nums'][num]:
                    hits[i] += 1
        e = {i for i, cnt in hits.items() if cnt >= 2}
    results_per['E'] = e
    
    union = set()
    for v in results_per.values():
        union.update(v)
    
    return {cids[j] for j in union}, results_per


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

def feats(s1nn, s1na, s1c, s1rn, s1ra, cnn, cna, cc, crn, cra):
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
    s1p, cp_ = get_postal(s1ra), get_postal(cra)
    pm = 1. if (s1p and cp_ and s1p==cp_) else 0.
    pa = 1. if (s1p and cp_) else 0.
    sn, cn = set(get_nums(s1ra)), set(get_nums(cra))
    numj = len(sn&cn)/max(len(sn|cn),1) if (sn or cn) else 0.
    cm = 1. if s1c==cc else 0.
    cts_ = (nts+ats)/2.; ctse_ = (ntse+atse)/2.
    nme = 1. if not s1nn else 0.; ade = 1. if not s1na else 0.
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
    pr("=" * 70)
    pr("FAST BASELINE: BLOCKING + CLASSIFIER ON DEV SAMPLE")
    pr("=" * 70)

    # [1] Load
    pr("\n[1] Load data")
    s1_full = load_source("train", 1)
    s2_full = load_source("train", 2)
    s3_full = load_source("train", 3)
    gt_full = load_ground_truth()

    # [2] Dev sample
    pr(f"\n[2] Sample {DEV_S1_SAMPLE:,} S1 entities")
    rng = np.random.RandomState(42)
    all_ids = list(gt_full.keys())
    rng.shuffle(all_ids)
    dev_ids = all_ids[:DEV_S1_SAMPLE]
    
    # Collect all true match IDs for dev entities
    true_match_ids = set()
    for sid in dev_ids:
        true_match_ids.update(gt_full[sid])
    pr(f"  True match IDs needed: {len(true_match_ids):,}")
    
    # Build candidate pool: true matches + random sample of S2/S3
    s2_ids_all = set(s2_full["entity_id"])
    s3_ids_all = set(s3_full["entity_id"])
    all_cand_ids = s2_ids_all | s3_ids_all
    
    # Random negative candidates (not in true match set)
    neg_pool = list(all_cand_ids - true_match_ids)
    rng.shuffle(neg_pool)
    neg_sample = set(neg_pool[:CAND_NEG_SAMPLE])
    
    keep_cand_ids = true_match_ids | neg_sample
    pr(f"  Candidate pool: {len(keep_cand_ids):,} (true={len(true_match_ids):,}, neg={len(neg_sample):,})")
    
    # Filter dataframes
    s1 = s1_full[s1_full["entity_id"].isin(dev_ids)].copy().reset_index(drop=True)
    s2 = s2_full[s2_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    s3 = s3_full[s3_full["entity_id"].isin(keep_cand_ids)].copy().reset_index(drop=True)
    gt = {k: gt_full[k] for k in dev_ids}
    
    pr(f"  Dev S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")
    
    # Free full data
    del s1_full, s2_full, s3_full, gt_full
    gc.collect()

    # [3] Normalize
    pr("\n[3] Normalize")
    add_norm_cols(s1)
    add_norm_cols(s2)
    add_norm_cols(s3)
    
    # [4] Train/val split
    pr(f"\n[4] Train/val split")
    dev_list = list(gt.keys())
    rng2 = np.random.RandomState(123)
    rng2.shuffle(dev_list)
    n_val = int(len(dev_list) * (1 - TRAIN_FRAC))
    val_ids = set(dev_list[:n_val])
    train_ids = set(dev_list[n_val:])
    gt_tr = {k:gt[k] for k in train_ids}
    gt_va = {k:gt[k] for k in val_ids}
    print_split_stats("Train", gt_tr)
    print_split_stats("Val", gt_va)
    
    # [5] Build blocking indices
    pr("\n[5] Build blocking indices")
    cdf = pd.concat([s2, s3], ignore_index=True)
    pr(f"  Candidates: {len(cdf):,}")
    cids, idx = build_indices(cdf)
    
    # [6] Block & measure recall
    pr("\n[6] Block validation set")
    val_s1 = s1[s1["entity_id"].isin(val_ids)].copy()
    
    val_cands = {}
    per_strategy_cands = {k: {} for k in ['A','B','C','D','E']}
    
    for _, row in tqdm(val_s1.iterrows(), total=len(val_s1), desc="  Blocking val"):
        eid = row["entity_id"]
        union_ids, per = block_one(row, cids, idx)
        val_cands[eid] = union_ids
        for k, v in per.items():
            per_strategy_cands[k][eid] = {cids[j] for j in v}
    
    pr("\n  === BLOCKING RECALL (validation) ===")
    pr(f"  {'Strategy':<12} {'Recall':>8} {'Pairs':>10} {'Miss':>6}")
    pr(f"  {'-'*40}")
    names = {'A':'Name exact','B':'Prefix-4','C':'Phonetic','D':'Postal','E':'Addr nums'}
    for k in ['A','B','C','D','E']:
        br = blocking_recall(per_strategy_cands[k], gt_va)
        pr(f"  {names[k]:<12} {br['recall_ceiling']:>8.4f} {br['total_candidates']:>10,} {br['missed_pairs']:>6,}")
    
    br_union = blocking_recall(val_cands, gt_va)
    pr(f"  {'UNION':<12} {br_union['recall_ceiling']:>8.4f} {br_union['total_candidates']:>10,} {br_union['missed_pairs']:>6,}")
    pr(f"\n  *** BLOCKING RECALL CEILING: {br_union['recall_ceiling']:.4f} ***")
    
    # [7] Block training set
    pr("\n[7] Block training set")
    tr_s1 = s1[s1["entity_id"].isin(train_ids)].copy()
    tr_cands = {}
    for _, row in tqdm(tr_s1.iterrows(), total=len(tr_s1), desc="  Blocking train"):
        eid = row["entity_id"]
        union_ids, _ = block_one(row, cids, idx)
        tr_cands[eid] = union_ids
    
    br_tr = blocking_recall(tr_cands, gt_tr)
    pr(f"  Train blocking recall: {br_tr['recall_ceiling']:.4f}")
    
    # [8] Build lookups
    pr("\n[8] Lookups")
    s1L = {}
    for _, r in s1.iterrows():
        s1L[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                r["business_name"],r["business_address"])
    cL = {}
    for _, r in cdf.iterrows():
        cL[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                               r["business_name"],r["business_address"])
    pr(f"  s1L={len(s1L):,}, cL={len(cL):,}")
    
    # [9] Generate pairs
    pr("\n[9] Generate labeled pairs")
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
    pr(f"  Train: {len(tr_pairs):,} (pos={sum(l for _,_,l in tr_pairs):,})")
    pr(f"  Val:   {len(va_pairs):,} (pos={sum(l for _,_,l in va_pairs):,})")
    
    # [10] Compute features
    pr("\n[10] Compute features")
    def to_Xy(pairs):
        X = np.zeros((len(pairs), len(FEAT_NAMES)), dtype=np.float32)
        y = np.zeros(len(pairs), dtype=np.int32)
        s1s, css = [], []
        for i,(sid,cid,lbl) in enumerate(pairs):
            s = s1L[sid]; c = cL[cid]
            X[i] = feats(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
            y[i] = lbl; s1s.append(sid); css.append(cid)
            if (i+1) % 50000 == 0: pr(f"    {i+1:,}/{len(pairs):,}")
        return X, y, s1s, css
    
    Xtr, ytr, _, _ = to_Xy(tr_pairs)
    pr(f"  Xtr: {Xtr.shape}, pos_rate={ytr.mean():.3f}")
    Xva, yva, _, _ = to_Xy(va_pairs)
    pr(f"  Xva: {Xva.shape}, pos_rate={yva.mean():.3f}")

    # [11] Train LightGBM
    pr("\n[11] Train LightGBM")
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=FEAT_NAMES)
    dval   = lgb.Dataset(Xva, label=yva, feature_name=FEAT_NAMES, reference=dtrain)
    mdl = lgb.train(
        {'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
         'num_leaves':127,'learning_rate':0.05,'feature_fraction':0.8,
         'bagging_fraction':0.8,'bagging_freq':5,'min_child_samples':50,
         'verbose':-1,'n_jobs':-1,'is_unbalance':True,'seed':42},
        dtrain, 1500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    pr(f"  Best iter: {mdl.best_iteration}")
    imp = sorted(zip(FEAT_NAMES, mdl.feature_importance('gain')), key=lambda x:-x[1])
    pr("\n  Top features:")
    for f,v in imp[:15]: pr(f"    {f}: {v:.0f}")

    # [12] Score ALL val candidates + threshold sweep
    pr("\n[12] Threshold calibration")
    avp = [(sid, cid) for sid, cs_ in val_cands.items() for cid in cs_ if cid in cL]
    pr(f"  Scoring {len(avp):,} val candidate pairs...")
    
    X_all = np.zeros((len(avp), len(FEAT_NAMES)), dtype=np.float32)
    all_s1v, all_cv = [], []
    for i,(sid,cid) in enumerate(avp):
        s = s1L[sid]; c = cL[cid]
        X_all[i] = feats(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
        all_s1v.append(sid); all_cv.append(cid)
        if (i+1) % 100000 == 0: pr(f"    {i+1:,}/{len(avp):,}")
    
    all_pr = mdl.predict(X_all)
    
    pr("\n  Sweeping thresholds...")
    best_f, best_t = 0., 0.5
    sweep_results = []
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = {s: set() for s in gt_va}
        for j in np.where(all_pr >= thr)[0]:
            s = all_s1v[j]
            if s in preds: preds[s].add(all_cv[j])
        r = f05_macro(preds, gt_va)
        sweep_results.append((thr, r))
        if r['f05'] > best_f: best_f = r['f05']; best_t = thr
    
    # Report best
    r = [x[1] for x in sweep_results if abs(x[0]-best_t)<0.005][0]
    pr(f"\n  *** BEST THRESHOLD: {best_t:.2f} ***")
    pr(f"      F_0.5 macro:     {r['f05']:.4f}")
    pr(f"      Singleton acc:   {r['sing_acc']:.4f}")
    pr(f"      Multi-match F05: {r['multi_f05']:.4f}")
    pr(f"      TP={r['tp']:,}, FP={r['fp']:,}, FN={r['fn']:,}")
    pr(f"      Pred empty={r['pred_empty']:,}, nonempty={r['pred_nonempty']:,}")
    
    # Top thresholds
    sweep_results.sort(key=lambda x: -x[1]['f05'])
    pr("\n  Top-10 thresholds:")
    for thr, r in sweep_results[:10]:
        pr(f"    thr={thr:.2f}: F05={r['f05']:.4f} sing={r['sing_acc']:.4f} multi={r['multi_f05']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    
    # [13] Error analysis sample
    pr("\n[13] Error analysis (sample)")
    preds_best = {s: set() for s in gt_va}
    for j in np.where(all_pr >= best_t)[0]:
        s = all_s1v[j]; 
        if s in preds_best: preds_best[s].add(all_cv[j])
    
    # FP examples
    fp_count = 0
    pr("\n  FALSE POSITIVES (wrong merges):")
    for sid in gt_va:
        tm = gt_va[sid]; pm = preds_best.get(sid, set())
        fps = pm - tm
        if fps and fp_count < 5:
            s = s1L[sid]
            pr(f"    S1={sid}: '{s[3]}' @ '{s[4]}'")
            for fpid in list(fps)[:2]:
                c = cL.get(fpid)
                if c: pr(f"      FP: {fpid}: '{c[3]}' @ '{c[4]}'")
            fp_count += 1
    
    # FN examples (blocked but not matched)
    fn_count = 0
    pr("\n  FALSE NEGATIVES (missed matches, in candidate set):")
    for sid in gt_va:
        tm = gt_va[sid]; pm = preds_best.get(sid, set())
        fns_in_cands = (tm - pm) & val_cands.get(sid, set())
        if fns_in_cands and fn_count < 5:
            s = s1L[sid]
            pr(f"    S1={sid}: '{s[3]}' @ '{s[4]}'")
            for fnid in list(fns_in_cands)[:2]:
                c = cL.get(fnid)
                if c: pr(f"      FN: {fnid}: '{c[3]}' @ '{c[4]}'")
            fn_count += 1
    
    # Save model
    mdl.save_model(os.path.join(CACHE_DIR, "baseline_model.txt"))
    
    pr(f"\n  Total time: {(time.time()-T0)/60:.1f} min")
    pr("\n  DONE. Next steps:")
    pr("    1. If blocking recall < 98%: add TF-IDF blocking channel")
    pr("    2. If FP rate high: add hard-negative mining")
    pr("    3. If val F0.5 acceptable: scale to full data for production run")


if __name__ == "__main__":
    main()
