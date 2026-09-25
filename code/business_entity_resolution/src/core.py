#!/usr/bin/env python3
"""
Core utilities for Business Entity Resolution Challenge.
- Complete UTF-8 safe I/O and logging
- Audited Macro F0.5 Evaluator
- Text normalization (country-agnostic)
- Blocking recall measurement
- Experiment tracking
"""
import os, sys, re, time, csv
from typing import Dict, Set, List, Tuple
import numpy as np
import pandas as pd

# Force UTF-8 on Windows stdout / stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR    = os.path.join(PROJECT_DIR, "..", "..",
                           "6ab10eb3b23ba_student_resource",
                           "student_resource", "dataset")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "..", "..", "output")
CACHE_DIR   = os.path.join(PROJECT_DIR, "cache")
EXP_LOG     = os.path.join(PROJECT_DIR, "experiments.csv")

for d in [OUTPUT_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Experiment Logging ─────────────────────────────────────────────────────
EXP_COLS = [
    "experiment_id", "date", "description", "blocking_recall",
    "val_entities", "val_pairs_scored", "threshold", "precision",
    "recall", "macro_f05", "singleton_acc", "multi_f05",
    "tp", "fp", "fn", "notes"
]

def log_experiment(row_dict: dict):
    """Append a row to experiments.csv."""
    file_exists = os.path.isfile(EXP_LOG)
    with open(EXP_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXP_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row_dict.get(col, "") for col in EXP_COLS})
    print(f"Logged experiment {row_dict.get('experiment_id')} to {EXP_LOG}", flush=True)

# ── Data loading ───────────────────────────────────────────────────────────
def load_source(split: str, num: int) -> pd.DataFrame:
    pre = "train" if split == "train" else "test"
    p = os.path.join(DATA_DIR, split, f"{pre}_source{num}.tsv")
    print(f"  Loading {os.path.basename(p)}...", end=" ", flush=True)
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, engine="c", encoding="utf-8")
    print(f"{len(df):,} rows", flush=True)
    return df

def load_ground_truth() -> Dict[str, Set[str]]:
    p = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    print(f"  Loading {os.path.basename(p)}...", end=" ", flush=True)
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    gt = {}
    s1_vals = df["source1_entity_id"].values
    m_vals  = df["matched_entity_ids"].values
    for s1, m in zip(s1_vals, m_vals):
        gt[s1] = set(m.split(",")) if m.strip() else set()
    print(f"{len(gt):,} rows", flush=True)
    return gt

# ── Text normalization (country-agnostic) ──────────────────────────────────
_LEGAL = re.compile(
    r'\bllc\b|\bllp\b|\binc\b|\bincorporated\b|\bcorp\b|\bcorporation\b'
    r'|\bco\b|\bcompany\b|\bltd\b|\blimited\b|\bplc\b|\blp\b'
    r'|\bgroup\b|\benterprises?\b|\bholdings?\b|\bpvt\b|\bprivate\b'
    r'|\bngo\b|\btrust\b|\bfoundation\b|\bsociety\b|\bopc\b'
    r'|\bsarl\b|\bsa\b|\bsas\b|\bsasu\b|\beurl\b|\bsnc\b|\bsca\b|\bsci\b|\bgie\b'
    r'|\b&\b|\band\b|\bet\b', re.IGNORECASE)
_PUNCT  = re.compile(r'[^\w\s]', re.UNICODE)
_SPACES = re.compile(r'\s+')
_NUMS   = re.compile(r'\d+')
_POSTAL = re.compile(r'\b(\d{5,6})\b')

_ADDR_SUBS = [
    (re.compile(r'\bstreet\b', re.I),    'st'),  (re.compile(r'\broad\b', re.I),      'rd'),
    (re.compile(r'\bavenue\b', re.I),    'ave'), (re.compile(r'\bboulevard\b', re.I), 'blvd'),
    (re.compile(r'\bdrive\b', re.I),     'dr'),  (re.compile(r'\blane\b', re.I),      'ln'),
    (re.compile(r'\bcourt\b', re.I),     'ct'),  (re.compile(r'\bplace\b', re.I),     'pl'),
    (re.compile(r'\bhighway\b', re.I),   'hwy'), (re.compile(r'\bsuite\b', re.I),     'ste'),
    (re.compile(r'\bapartment\b', re.I), 'apt'), (re.compile(r'\bbuilding\b', re.I),  'bldg'),
    (re.compile(r'\bfloor\b', re.I),     'fl'),
]

def norm_name(name: str) -> str:
    if not name: return ""
    s = str(name).lower().strip()
    s = _LEGAL.sub(' ', s)
    s = _PUNCT.sub(' ', s)
    return _SPACES.sub(' ', s).strip()

def norm_name_sorted(name: str) -> str:
    n = norm_name(name)
    return ' '.join(sorted(n.split())) if n else ""

def norm_addr(addr: str) -> str:
    if not addr: return ""
    s = str(addr).lower().strip()
    for p, r in _ADDR_SUBS: s = p.sub(r, s)
    s = _PUNCT.sub(' ', s)
    return _SPACES.sub(' ', s).strip()

def get_postal(addr: str) -> str:
    if not addr: return ""
    m = _POSTAL.findall(str(addr))
    return m[-1] if m else ""

def get_nums(text: str) -> list:
    if not text: return []
    return _NUMS.findall(str(text))

def add_norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add precomputed normalized columns to a source dataframe."""
    df["_nn"]  = df["business_name"].apply(norm_name)
    df["_nns"] = df["business_name"].apply(norm_name_sorted)
    df["_na"]  = df["business_address"].apply(norm_addr)
    df["_pc"]  = df["business_address"].apply(get_postal)
    return df

# ── Audited Macro F0.5 Evaluator ───────────────────────────────────────────
def f05_per_entity(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (f0.5, precision, recall) for a single source1 entity.
    
    Audit against rules:
    - true_set is empty (singleton):
        - pred_set is empty -> score = 1.0, precision=1.0, recall=1.0
        - pred_set is non-empty -> score = 0.0, precision=0.0, recall=0.0
    - true_set is non-empty:
        - pred_set is empty -> score = 0.0, precision=0.0, recall=0.0
        - pred_set is non-empty:
            tp = |true & pred|
            precision = tp / |pred|
            recall = tp / |true|
            f0.5 = (1.25 * P * R) / (0.25 * P + R) if (P + R) > 0 else 0.0
    """
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0  # Precision is 0, F0.5 is 0
    if not pred_set:
        return 0.0, 1.0, 0.0      # Recall is 0, F0.5 is 0
    
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    
    prec = tp / len(pred_set)
    rec  = tp / len(true_set)
    f05  = (1.25 * prec * rec) / (0.25 * prec + rec)
    return f05, prec, rec

def f05_macro(preds: Dict[str, Set[str]],
              truth: Dict[str, Set[str]]) -> dict:
    """
    Macro-averaged F0.5 across all source1 entities in `truth`.
    Also computes global micro stats (TP, FP, FN) and separate singleton/multi breakdowns.
    """
    scores, precs, recs = [], [], []
    sing_scores, multi_scores = [], []
    tp_tot = fp_tot = fn_tot = 0
    
    for s1, true_set in truth.items():
        pred_set = preds.get(s1, set())
        sc, p, r = f05_per_entity(true_set, pred_set)
        scores.append(sc)
        precs.append(p)
        recs.append(r)
        
        if not true_set:
            sing_scores.append(sc)
            if pred_set:
                fp_tot += len(pred_set)
        else:
            multi_scores.append(sc)
            tp = len(true_set & pred_set)
            tp_tot += tp
            fp_tot += len(pred_set - true_set)
            fn_tot += len(true_set - pred_set)
            
    n_pred_empty = sum(1 for s1 in truth if not preds.get(s1, set()))
    n_pred_nonempty = len(truth) - n_pred_empty
    
    micro_prec = tp_tot / max(tp_tot + fp_tot, 1)
    micro_rec  = tp_tot / max(tp_tot + fn_tot, 1)
    
    return {
        'macro_f05':     float(np.mean(scores)) if scores else 0.0,
        'macro_prec':    float(np.mean(precs)) if precs else 0.0,
        'macro_rec':     float(np.mean(recs)) if recs else 0.0,
        'singleton_acc': float(np.mean(sing_scores)) if sing_scores else 0.0,
        'multi_f05':     float(np.mean(multi_scores)) if multi_scores else 0.0,
        'n_entities':    len(scores),
        'n_singletons':  len(sing_scores),
        'n_multi':       len(multi_scores),
        'tp':            tp_tot,
        'fp':            fp_tot,
        'fn':            fn_tot,
        'micro_prec':    micro_prec,
        'micro_rec':     micro_rec,
        'pred_empty':    n_pred_empty,
        'pred_nonempty': n_pred_nonempty,
    }

def blocking_recall(candidates: Dict[str, Set[str]],
                    truth: Dict[str, Set[str]]) -> dict:
    """Measure blocking recall ceiling."""
    ttrue = tfound = 0
    missed_entities = 0
    for s1, tm in truth.items():
        if not tm: continue
        ttrue += len(tm)
        found = len(tm & candidates.get(s1, set()))
        tfound += found
        if found < len(tm):
            missed_entities += 1
    rc = tfound / max(ttrue, 1)
    total_cands = sum(len(v) for v in candidates.values())
    return {
        'recall_ceiling': rc,
        'true_pairs':     ttrue,
        'found_pairs':    tfound,
        'missed_pairs':   ttrue - tfound,
        'missed_entities': missed_entities,
        'total_candidates': total_cands,
        'avg_cands':      total_cands / max(len(candidates), 1),
    }

# ── Entity-level train/val split ──────────────────────────────────────────
def entity_split(gt: Dict[str, Set[str]], val_frac: float = 0.2,
                 seed: int = 42, max_val: int = None):
    """Stratified entity-level split preserving singleton ratio."""
    rng = np.random.RandomState(seed)
    all_ids = list(gt.keys())
    rng.shuffle(all_ids)
    sings  = [x for x in all_ids if not gt[x]]
    nonsings = [x for x in all_ids if gt[x]]
    
    n_val = int(len(all_ids) * val_frac)
    if max_val: n_val = min(n_val, max_val)
    sf = len(sings) / len(all_ids)
    vs = int(n_val * sf)
    vn = n_val - vs
    
    val_ids   = set(sings[:vs] + nonsings[:vn])
    train_ids = set(all_ids) - val_ids
    
    return ({k: gt[k] for k in train_ids},
            {k: gt[k] for k in val_ids})

def print_split_stats(name, gt_split):
    ns = sum(1 for v in gt_split.values() if not v)
    nm = len(gt_split) - ns
    print(f"  {name}: {len(gt_split):,} entities "
          f"({ns:,} singletons, {nm:,} non-singletons)", flush=True)
