#!/usr/bin/env python3
"""
Business Entity Resolution Pipeline
====================================
Scalable pipeline for 2M S1 × 10M S2/S3 entity resolution.

Architecture:
  Phase 1 — BLOCKING (union of strategies, target ≥98% recall ceiling)
    a) Normalized-sorted name exact match
    b) Double Metaphone phonetic keys
    c) Postal code exact match
    d) First-3-chars-of-name + country prefix (high recall, cheap)
    e) TF-IDF char n-gram (name only) via sklearn sparse kNN — language-agnostic safety net

  Phase 2 — FEATURE ENGINEERING (30 similarity features per pair)
  Phase 3 — LIGHTGBM CLASSIFIER
  Phase 4 — F_0.5 THRESHOLD CALIBRATION

All logic is country-agnostic. No hardcoded country filters.
Zero external data lookups.
"""

import os, re, sys, gc, time, math, warnings, heapq
from collections import defaultdict, Counter
from typing import Dict, Set, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
from metaphone import doublemetaphone
from tqdm import tqdm
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
BASE_DIR = os.path.join(PROJECT_DIR, "..", "..",
                        "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "..", "..", "output")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

# ============================================================================
# CONFIG
# ============================================================================
TRAIN_SAMPLE  = 80000
VAL_SAMPLE    = 20000
NEG_RATIO     = 5
TFIDF_TOP_K   = 10            # kNN neighbors from TF-IDF blocking
TFIDF_NAME_MAX_FEAT = 50000   # max TF-IDF features (reduced for 16GB RAM)
TFIDF_BATCH   = 10000         # batch size for kNN queries
BLOCKING_BATCH = 50000        # batch size for hash-based blocking

LGBM_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'boosting_type': 'gbdt', 'num_leaves': 127,
    'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'verbose': -1, 'n_jobs': -1,
    'is_unbalance': True, 'seed': 42,
}

# ============================================================================
# TEXT NORMALIZATION (country-agnostic)
# ============================================================================
_LEGAL_RE = re.compile(
    r'\bllc\b|\bllp\b|\binc\b|\bincorporated\b|\bcorp\b|\bcorporation\b'
    r'|\bco\b|\bcompany\b|\bltd\b|\blimited\b|\bplc\b|\blp\b'
    r'|\bgroup\b|\benterprises?\b|\bholdings?\b|\bpvt\b|\bprivate\b'
    r'|\bngo\b|\btrust\b|\bfoundation\b|\bsociety\b|\bopc\b'
    r'|\bsarl\b|\bsa\b|\bsas\b|\bsasu\b|\beurl\b|\bsnc\b|\bsca\b|\bsci\b|\bgie\b'
    r'|\b&\b|\band\b|\bet\b', re.IGNORECASE)
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_SPACES = re.compile(r'\s+')
_NUMS = re.compile(r'\d+')
_POSTAL = re.compile(r'\b(\d{5,6})\b')

_ADDR_SUBS = [
    (re.compile(r'\bstreet\b'), 'st'), (re.compile(r'\broad\b'), 'rd'),
    (re.compile(r'\bavenue\b'), 'ave'), (re.compile(r'\bboulevard\b'), 'blvd'),
    (re.compile(r'\bdrive\b'), 'dr'), (re.compile(r'\blane\b'), 'ln'),
    (re.compile(r'\bcourt\b'), 'ct'), (re.compile(r'\bplace\b'), 'pl'),
    (re.compile(r'\bhighway\b'), 'hwy'), (re.compile(r'\bsuite\b'), 'ste'),
    (re.compile(r'\bapartment\b'), 'apt'), (re.compile(r'\bbuilding\b'), 'bldg'),
    (re.compile(r'\bfloor\b'), 'fl'),
]

def nn(name):
    """Normalize business name."""
    if not name: return ""
    s = str(name).lower().strip()
    s = _LEGAL_RE.sub(' ', s)
    s = _PUNCT.sub(' ', s)
    return _SPACES.sub(' ', s).strip()

def nns(name):
    """Normalize name + sort tokens."""
    n = nn(name)
    return ' '.join(sorted(n.split())) if n else ""

def na(addr):
    """Normalize address."""
    if not addr: return ""
    s = str(addr).lower().strip()
    for p, r in _ADDR_SUBS:
        s = p.sub(r, s)
    s = _PUNCT.sub(' ', s)
    return _SPACES.sub(' ', s).strip()

def postal(addr):
    if not addr: return ""
    m = _POSTAL.findall(str(addr))
    return m[-1] if m else ""

def nums(text):
    if not text: return []
    return _NUMS.findall(str(text))

def pkeys(name):
    """Double Metaphone keys for tokens ≥ 2 chars."""
    n = nn(name)
    if not n: return set()
    keys = set()
    for t in n.split():
        if len(t) < 2: continue
        try:
            p, s = doublemetaphone(t)
            if p: keys.add(p)
            if s: keys.add(s)
        except: pass
    return keys

def name_prefix(name, k=3):
    """First k chars of normalized name (for prefix blocking)."""
    n = nn(name)
    return n[:k] if len(n) >= k else n

# ============================================================================
# DATA LOADING
# ============================================================================
def load_src(split, num):
    pre = "train" if split == "train" else "test"
    p = os.path.join(BASE_DIR, split, f"{pre}_source{num}.tsv")
    print(f"  Loading {os.path.basename(p)}...", end=" ", flush=True)
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, engine='c')
    print(f"{len(df):,} rows")
    return df

def load_gt():
    p = os.path.join(BASE_DIR, "train", "train_ground_truth.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, r in df.iterrows():
        s1 = r["source1_entity_id"]
        m = r["matched_entity_ids"]
        gt[s1] = set(m.split(",")) if m.strip() else set()
    return gt

def add_norms(df):
    """Add precomputed normalized columns."""
    df["_nn"]  = df["business_name"].apply(nn)
    df["_nns"] = df["business_name"].apply(nns)
    df["_na"]  = df["business_address"].apply(na)
    df["_pc"]  = df["business_address"].apply(postal)
    df["_pfx"] = df["business_name"].apply(name_prefix)
    return df

# ============================================================================
# BLOCKING
# ============================================================================

class Blocker:
    """
    Union blocker: hash indices + TF-IDF kNN safety net.
    
    Hash strategies are O(1) per query → scale to millions.
    TF-IDF kNN uses sklearn's efficient sparse brute-force → batch queries.
    """

    def __init__(self):
        self.cand_ids = None
        # Hash indices: key → list of candidate indices
        self.idx_name = {}
        self.idx_phone = {}
        self.idx_postal = {}
        self.idx_pfx = {}
        # TF-IDF kNN
        self.tfidf_vec = None
        self.tfidf_nn = None

    def build(self, cands: pd.DataFrame):
        """Build all blocking indices on the combined S2+S3 dataframe."""
        t0 = time.time()
        n = len(cands)
        self.cand_ids = cands["entity_id"].values
        print(f"\n  Building blocking indices on {n:,} candidates...")

        nns_v = cands["_nns"].values
        pc_v  = cands["_pc"].values
        pfx_v = cands["_pfx"].values
        raw_n = cands["business_name"].values
        nn_v  = cands["_nn"].values

        # 1) Exact normalized-sorted name
        print("    [1/5] Name exact index...", end=" ", flush=True)
        idx = defaultdict(list)
        for i, v in enumerate(nns_v):
            if v: idx[v].append(i)
        self.idx_name = dict(idx)
        print(f"{len(self.idx_name):,} keys")

        # 2) Phonetic
        print("    [2/5] Phonetic index...", flush=True)
        idx = defaultdict(list)
        for i in tqdm(range(n), desc="      Phonetic", mininterval=15):
            for k in pkeys(raw_n[i]):
                idx[k].append(i)
        self.idx_phone = dict(idx)
        print(f"      {len(self.idx_phone):,} keys")

        # 3) Postal code
        print("    [3/5] Postal code index...", end=" ", flush=True)
        idx = defaultdict(list)
        for i, v in enumerate(pc_v):
            if v: idx[v].append(i)
        self.idx_postal = dict(idx)
        print(f"{len(self.idx_postal):,} keys")

        # 4) Name prefix (first 3 chars)
        print("    [4/5] Name prefix index...", end=" ", flush=True)
        idx = defaultdict(list)
        for i, v in enumerate(pfx_v):
            if v: idx[v].append(i)
        self.idx_pfx = dict(idx)
        print(f"{len(self.idx_pfx):,} keys")

        # 5) TF-IDF on name (char n-grams) + kNN
        print("    [5/5] TF-IDF name index...", flush=True)
        self.tfidf_vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            max_features=TFIDF_NAME_MAX_FEAT,
            sublinear_tf=True, dtype=np.float32,
        )
        tfidf_mat = self.tfidf_vec.fit_transform(nn_v.astype(str))
        print(f"      TF-IDF shape: {tfidf_mat.shape}, nnz: {tfidf_mat.nnz:,}")

        self.tfidf_nn = NearestNeighbors(
            n_neighbors=min(TFIDF_TOP_K, n),
            metric='cosine', algorithm='brute', n_jobs=-1,
        )
        self.tfidf_nn.fit(tfidf_mat)
        print(f"    All indices built in {time.time()-t0:.0f}s")

    def _hash_candidates(self, nns_val, raw_name, pc_val, pfx_val, na_val):
        """Get candidates from all hash-based strategies."""
        cands = set()

        # 1) Exact name
        if nns_val:
            bucket = self.idx_name.get(nns_val)
            if bucket: cands.update(bucket)

        # 2) Phonetic
        for k in pkeys(raw_name):
            bucket = self.idx_phone.get(k)
            if bucket and len(bucket) < 10000:
                cands.update(bucket)

        # 3) Postal code
        if pc_val:
            bucket = self.idx_postal.get(pc_val)
            if bucket: cands.update(bucket)

        # 4) Name prefix — only use if name is ≥ 3 chars
        if pfx_val and len(pfx_val) >= 3:
            bucket = self.idx_pfx.get(pfx_val)
            if bucket and len(bucket) < 20000:
                cands.update(bucket)

        return cands

    def block_all(self, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Block all S1 entities. Returns {s1_id: set(candidate_ids)}."""
        n = len(s1_df)
        print(f"\n  Blocking {n:,} S1 entities...")

        ids   = s1_df["entity_id"].values
        rn    = s1_df["business_name"].values
        nns_v = s1_df["_nns"].values
        nn_v  = s1_df["_nn"].values
        pc_v  = s1_df["_pc"].values
        pfx_v = s1_df["_pfx"].values
        na_v  = s1_df["_na"].values

        # Step A: hash-based blocking (fast)
        result = {}
        for i in tqdm(range(n), desc="    Hash blocking", mininterval=15):
            idx_set = self._hash_candidates(nns_v[i], rn[i], pc_v[i], pfx_v[i], na_v[i])
            result[ids[i]] = idx_set

        # Step B: TF-IDF kNN blocking in batches
        print("    TF-IDF kNN blocking...", flush=True)
        for start in tqdm(range(0, n, TFIDF_BATCH), desc="    TF-IDF batches"):
            end = min(start + TFIDF_BATCH, n)
            batch_nn = nn_v[start:end].astype(str)
            q = self.tfidf_vec.transform(batch_nn)
            _, knn_idx = self.tfidf_nn.kneighbors(q)
            for i in range(end - start):
                eid = ids[start + i]
                result[eid].update(knn_idx[i].tolist())

        # Convert index sets to ID sets
        final = {}
        for eid, idx_set in result.items():
            final[eid] = {self.cand_ids[j] for j in idx_set}

        total = sum(len(v) for v in final.values())
        nonempty = sum(1 for v in final.values() if v)
        print(f"    Candidate pairs: {total:,}")
        print(f"    Avg/S1: {total/max(n,1):.1f}")
        print(f"    S1 w/ ≥1 cand: {nonempty:,}/{n:,}")
        return final

# ============================================================================
# FEATURES (30 similarity metrics)
# ============================================================================
FEAT_NAMES = [
    'name_jaccard', 'name_tri_jacc', 'name_lev', 'name_tsort', 'name_tset',
    'name_partial', 'name_wr', 'name_exact', 'name_exact_sorted',
    'name_len_d', 'name_len_r', 'name_tok_d',
    'addr_jaccard', 'addr_tri_jacc', 'addr_lev', 'addr_tsort', 'addr_tset',
    'addr_partial', 'addr_len_d', 'addr_len_r', 'addr_tok_d',
    'post_match', 'post_avail', 'num_jacc',
    'country_m', 'comb_tsort', 'comb_tset', 'name_empty', 'addr_empty',
    'phone_m',
]

def _tri(s):
    return set(s[i:i+3] for i in range(max(0, len(s)-2)))

def feats(s1nn, s1na, s1c, s1rn, s1ra, cnn, cna, cc, crn, cra):
    """30 features for one pair."""
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
    s1p, cp_ = postal(s1ra), postal(cra)
    pm = 1. if (s1p and cp_ and s1p==cp_) else 0.
    pa = 1. if (s1p and cp_) else 0.
    sn, cn = set(nums(s1ra)), set(nums(cra))
    numj = len(sn&cn)/max(len(sn|cn),1) if (sn or cn) else 0.
    # Cross
    cm = 1. if s1c==cc else 0.
    cts_ = (nts+ats)/2.
    ctse_ = (ntse+atse)/2.
    nme = 1. if not s1nn else 0.
    ade = 1. if not s1na else 0.
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

# ============================================================================
# EVALUATION
# ============================================================================
def f05_macro(preds, truth):
    scores, sing, multi = [], [], []
    for s1 in truth:
        t = truth[s1]; p = preds.get(s1, set())
        if not t:
            sc = 1. if not p else 0.; sing.append(sc)
        else:
            if not p: sc = 0.
            else:
                tp = len(t&p); pr = tp/len(p); rc = tp/len(t)
                sc = 1.25*pr*rc/(0.25*pr+rc) if (pr+rc)>0 else 0.
            multi.append(sc)
        scores.append(sc)
    return {'f05': np.mean(scores), 'sing': np.mean(sing) if sing else 0.,
            'multi': np.mean(multi) if multi else 0.,
            'n': len(scores), 'ns': len(sing), 'nm': len(multi)}

# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    T0 = time.time()

    # ===== LOAD =====
    print("="*70); print("STEP 1: LOAD DATA"); print("="*70)
    s1 = load_src("train", 1)
    s2 = load_src("train", 2)
    s3 = load_src("train", 3)
    gt = load_gt()
    ns = sum(1 for v in gt.values() if not v)
    print(f"  GT: {len(gt):,}, singletons: {ns:,} ({100*ns/len(gt):.1f}%)")

    # ===== NORMALIZE =====
    print("\n"+"="*70); print("STEP 2: NORMALIZE"); print("="*70)
    for lbl, df in [("S1",s1),("S2",s2),("S3",s3)]:
        print(f"  {lbl}:"); add_norms(df)

    # ===== SPLIT =====
    print("\n"+"="*70); print("STEP 3: TRAIN/VAL SPLIT"); print("="*70)
    rng = np.random.RandomState(42)
    ids_all = list(gt.keys()); rng.shuffle(ids_all)
    sids = [x for x in ids_all if not gt[x]]
    nids = [x for x in ids_all if gt[x]]
    sf = len(sids)/len(ids_all)
    vs = int(VAL_SAMPLE*sf); vn = VAL_SAMPLE-vs
    val_ids = set(sids[:vs] + nids[:vn])
    ts = int(TRAIN_SAMPLE*sf); tn = TRAIN_SAMPLE-ts
    rs = [x for x in sids if x not in val_ids]
    rn_ = [x for x in nids if x not in val_ids]
    train_ids = set(rs[:ts] + rn_[:tn])
    gt_tr = {k:gt[k] for k in train_ids}
    gt_va = {k:gt[k] for k in val_ids}
    print(f"  Train: {len(gt_tr):,} (sing={sum(1 for v in gt_tr.values() if not v):,})")
    print(f"  Val:   {len(gt_va):,} (sing={sum(1 for v in gt_va.values() if not v):,})")

    # ===== BLOCKING INDEX =====
    print("\n"+"="*70); print("STEP 4: BUILD BLOCKING INDEX"); print("="*70)
    cdf = pd.concat([s2, s3], ignore_index=True)
    print(f"  S2+S3 candidates: {len(cdf):,}")
    blk = Blocker()
    blk.build(cdf)

    # ===== BLOCK VALIDATION =====
    print("\n"+"="*70); print("STEP 5: BLOCK VALIDATION SET"); print("="*70)
    val_s1 = s1[s1["entity_id"].isin(val_ids)].copy()
    vc = blk.block_all(val_s1)

    ttrue = tfound = 0
    for sid, tm in gt_va.items():
        if not tm: continue
        ttrue += len(tm)
        tfound += len(tm & vc.get(sid, set()))
    rc = tfound/max(ttrue,1)
    print(f"\n  *** BLOCKING RECALL CEILING: {rc:.4f} ({100*rc:.2f}%) ***")
    print(f"      True: {ttrue:,}, Found: {tfound:,}, Missed: {ttrue-tfound:,}")

    # ===== BLOCK TRAINING =====
    print("\n"+"="*70); print("STEP 6: BLOCK TRAINING SET"); print("="*70)
    tr_s1 = s1[s1["entity_id"].isin(train_ids)].copy()
    tc = blk.block_all(tr_s1)

    ttrue_t = tfound_t = 0
    for sid, tm in gt_tr.items():
        if not tm: continue
        ttrue_t += len(tm)
        tfound_t += len(tm & tc.get(sid, set()))
    print(f"    Train blocking recall: {tfound_t/max(ttrue_t,1):.4f}")

    # ===== LOOKUPS =====
    print("\n"+"="*70); print("STEP 7: BUILD LOOKUPS"); print("="*70)
    needed_s1 = train_ids | val_ids
    s1L = {}
    for _, r in s1.iterrows():
        if r["entity_id"] in needed_s1:
            s1L[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                    r["business_name"],r["business_address"])
    needed_c = set()
    for cs_ in tc.values(): needed_c.update(cs_)
    for cs_ in vc.values(): needed_c.update(cs_)
    cL = {}
    for _, r in cdf.iterrows():
        if r["entity_id"] in needed_c:
            cL[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                   r["business_name"],r["business_address"])
    print(f"  S1 lookup: {len(s1L):,}")
    print(f"  Cand lookup: {len(cL):,}")

    # ===== PAIRS & FEATURES =====
    print("\n"+"="*70); print("STEP 8: PAIRS & FEATURES"); print("="*70)
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

    tp = gen_pairs(tc, gt_tr, NEG_RATIO)
    vp = gen_pairs(vc, gt_va, NEG_RATIO)
    print(f"  Train pairs: {len(tp):,} (pos={sum(l for _,_,l in tp):,})")
    print(f"  Val pairs:   {len(vp):,} (pos={sum(l for _,_,l in vp):,})")

    def to_Xy(pairs):
        X = np.zeros((len(pairs), len(FEAT_NAMES)), dtype=np.float32)
        y = np.zeros(len(pairs), dtype=np.int32)
        s1s, css = [], []
        for i,(sid,cid,lbl) in enumerate(tqdm(pairs, desc="  Features", mininterval=15)):
            s = s1L[sid]; c = cL[cid]
            X[i] = feats(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
            y[i] = lbl; s1s.append(sid); css.append(cid)
        return X, y, s1s, css

    print("\n  Training features...")
    Xtr, ytr, _, _ = to_Xy(tp)
    print("  Validation features...")
    Xva, yva, vs1, vcs = to_Xy(vp)

    # ===== TRAIN =====
    print("\n"+"="*70); print("STEP 9: TRAIN LIGHTGBM"); print("="*70)
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=FEAT_NAMES)
    dval   = lgb.Dataset(Xva, label=yva, feature_name=FEAT_NAMES, reference=dtrain)
    mdl = lgb.train(LGBM_PARAMS, dtrain, 1500, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"  Best iter: {mdl.best_iteration}")
    imp = sorted(zip(FEAT_NAMES, mdl.feature_importance('gain')), key=lambda x:-x[1])
    print("\n  Top features:"); 
    for f,v in imp[:15]: print(f"    {f}: {v:.0f}")

    # ===== THRESHOLD =====
    print("\n"+"="*70); print("STEP 10: THRESHOLD CALIBRATION"); print("="*70)
    print("  Scoring all val candidates...")
    avp = []
    for sid, cs_ in vc.items():
        for cid in cs_:
            if cid in cL: avp.append((sid, cid))
    print(f"    {len(avp):,} pairs")

    # Chunked scoring
    chunk = 500000
    all_pr, all_s1, all_c = [], [], []
    for st in range(0, len(avp), chunk):
        en = min(st+chunk, len(avp))
        ch = avp[st:en]
        Xch = np.zeros((len(ch), len(FEAT_NAMES)), dtype=np.float32)
        for i,(sid,cid) in enumerate(tqdm(ch, desc=f"  Chunk {st//chunk+1}", mininterval=15)):
            s = s1L[sid]; c = cL[cid]
            Xch[i] = feats(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
        pr = mdl.predict(Xch)
        all_pr.extend(pr.tolist())
        all_s1.extend([p[0] for p in ch])
        all_c.extend([p[1] for p in ch])
        del Xch; gc.collect()

    all_pr = np.array(all_pr)
    print(f"    Scored {len(all_pr):,} pairs")

    print("\n  Sweeping thresholds...")
    best_f, best_t = 0., 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = {s: set() for s in gt_va}
        for j in np.where(all_pr >= thr)[0]:
            s = all_s1[j]
            if s in preds: preds[s].add(all_c[j])
        r = f05_macro(preds, gt_va)
        if r['f05'] > best_f: best_f = r['f05']; best_t = thr

    # Report
    preds_b = {s: set() for s in gt_va}
    for j in np.where(all_pr >= best_t)[0]:
        s = all_s1[j]
        if s in preds_b: preds_b[s].add(all_c[j])
    r = f05_macro(preds_b, gt_va)
    print(f"\n  *** BEST THRESHOLD: {best_t:.2f} ***")
    print(f"      F_0.5 macro:     {r['f05']:.4f}")
    print(f"      Singleton acc:   {r['sing']:.4f}")
    print(f"      Multi-match F05: {r['multi']:.4f}")
    print(f"      N={r['n']}, Sing={r['ns']}, Multi={r['nm']}")

    # ===== CROSS-COUNTRY =====
    print("\n"+"="*70); print("STEP 11: CROSS-COUNTRY CHECK"); print("="*70)
    for ho in ["US", "India"]:
        ho_ids = set(s1[s1["country"]==ho]["entity_id"]) & set(gt.keys())
        if not ho_ids: continue
        sample = list(ho_ids)[:3000]
        hgt = {k:gt[k] for k in sample}
        hs1 = s1[s1["entity_id"].isin(sample)].copy()
        hc = blk.block_all(hs1)
        hp = []
        for sid, cs_ in hc.items():
            for cid in cs_:
                if cid in cL: hp.append((sid, cid))
        if not hp: continue
        # Need s1L entries for holdout
        for sid in sample:
            if sid not in s1L:
                row = s1[s1["entity_id"]==sid].iloc[0]
                s1L[sid] = (row["_nn"],row["_na"],row["country"],
                            row["business_name"],row["business_address"])
        Xh = np.zeros((len(hp), len(FEAT_NAMES)), dtype=np.float32)
        hs, hcs_ = [], []
        for i,(sid,cid) in enumerate(hp):
            s = s1L.get(sid); c = cL.get(cid)
            if not s or not c: continue
            Xh[i] = feats(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4])
            hs.append(sid); hcs_.append(cid)
        pr_h = mdl.predict(Xh[:len(hs)])
        pred_h = {s: set() for s in hgt}
        for j,p in enumerate(pr_h):
            if p >= best_t and hs[j] in pred_h:
                pred_h[hs[j]].add(hcs_[j])
        hr = f05_macro(pred_h, hgt)
        print(f"  {ho:6s}: F05={hr['f05']:.4f}, sing={hr['sing']:.4f}, multi={hr['multi']:.4f}")

    # ===== RETRAIN FULL =====
    print("\n"+"="*70); print("STEP 12: RETRAIN ON ALL DATA"); print("="*70)
    Xfull = np.concatenate([Xtr, Xva])
    yfull = np.concatenate([ytr, yva])
    dfull = lgb.Dataset(Xfull, label=yfull, feature_name=FEAT_NAMES)
    fmdl = lgb.train(LGBM_PARAMS, dfull, num_boost_round=mdl.best_iteration)
    fmdl.save_model(os.path.join(CACHE_DIR, "final_model.txt"))
    print(f"  Model saved. Iterations: {mdl.best_iteration}")

    # Free mem
    del Xtr, Xva, Xfull, ytr, yva, yfull, dtrain, dval, dfull
    del s1L, cL, tc, vc, all_pr, cdf
    del s1, s2, s3
    gc.collect()

    # ===== TEST PREDICTIONS =====
    print("\n"+"="*70); print("STEP 13: TEST PREDICTIONS"); print("="*70)
    s1t = load_src("test", 1)
    s2t = load_src("test", 2)
    s3t = load_src("test", 3)
    for lbl, df in [("S1",s1t),("S2",s2t),("S3",s3t)]:
        print(f"  Norm {lbl}:"); add_norms(df)

    tcdf = pd.concat([s2t, s3t], ignore_index=True)
    print(f"  Test candidates: {len(tcdf):,}")

    tblk = Blocker()
    tblk.build(tcdf)

    # Lookups
    print("  Building test lookups...")
    s1tL = {}
    for _, r in s1t.iterrows():
        s1tL[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                 r["business_name"],r["business_address"])
    ctL = {}
    for _, r in tcdf.iterrows():
        ctL[r["entity_id"]] = (r["_nn"],r["_na"],r["country"],
                                r["business_name"],r["business_address"])
    print(f"  S1t: {len(s1tL):,}, Ct: {len(ctL):,}")

    test_cands = tblk.block_all(s1t)

    # Score
    print("\n  Scoring test candidates...")
    test_res = {}
    test_co = {}
    test_ids = s1t["entity_id"].values

    for sid in tqdm(test_ids, desc="  Scoring", mininterval=15):
        cs_ = test_cands.get(sid, set())
        test_co[sid] = cs_
        if not cs_:
            test_res[sid] = set(); continue
        s = s1tL[sid]
        fl, cl_ = [], []
        for cid in cs_:
            c = ctL.get(cid)
            if not c: continue
            fl.append(feats(s[0],s[1],s[2],s[3],s[4], c[0],c[1],c[2],c[3],c[4]))
            cl_.append(cid)
        if not fl:
            test_res[sid] = set(); continue
        X = np.array(fl)
        pr = fmdl.predict(X)
        test_res[sid] = {cid for cid,p in zip(cl_,pr) if p >= best_t}

    # ===== OUTPUT =====
    print("\n"+"="*70); print("STEP 14: WRITE OUTPUT"); print("="*70)
    mp = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in test_ids:
            m = test_res.get(sid, set())
            f.write(f"{sid}\t{','.join(sorted(m))}\n")
    print(f"  Written: {mp}")

    cp = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    with open(cp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in test_ids:
            c = test_co.get(sid, set())
            f.write(f"{sid}\t{','.join(sorted(c))}\n")
    print(f"  Written: {cp}")

    tm_ = sum(len(v) for v in test_res.values())
    emp = sum(1 for v in test_res.values() if not v)
    print(f"\n  Matched: {tm_:,}")
    print(f"  Singletons: {emp:,}/{len(test_res):,}")
    print(f"  Time: {(time.time()-T0)/60:.1f} min")
    print("\n  DONE!")


if __name__ == "__main__":
    main()
