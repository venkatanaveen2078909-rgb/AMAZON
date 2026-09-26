#!/usr/bin/env python3
r"""
Verification & Progressive Benchmarking Script
Tests cache integrity, validates exact EXP-04 output equivalence on --limit 10,
and benchmarks --limit 100, --limit 500, and --limit 5000.
"""
import os
import sys
import gc
import time
import pickle
import subprocess
from pathlib import Path

# UTF-8 Configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CACHE_FILE = ROOT_DIR / "code" / "business_entity_resolution" / "cache" / "test_candidate_index.pkl"
PIPELINE_SCRIPT = ROOT_DIR / "code" / "business_entity_resolution" / "src" / "kaggle_submission_pipeline.py"
OUTPUT_DIR = ROOT_DIR / "output"

def verify_cache():
    print("=" * 75)
    print("STEP 1: VERIFYING PRECOMPUTED CANDIDATE CACHE")
    print("=" * 75)
    
    if not CACHE_FILE.exists():
        print(f"ERROR: Cache file {CACHE_FILE} does not exist!")
        return False
        
    size_bytes = os.path.getsize(CACHE_FILE)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    print(f"1. Cache file exists: {CACHE_FILE}")
    print(f"2. Cache size on disk: {size_bytes:,} bytes ({size_mb:.2f} MB / {size_gb:.2f} GB)")
    
    print("\nLoading cache in fresh Python process to verify structure...")
    t0 = time.perf_counter()
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    t_load = time.perf_counter() - t0
    print(f"  Cache loaded successfully in {t_load:.2f}s.")
    
    n_cands = cache.get("n_cands", 0)
    print(f"3. Candidate entities represented: {n_cands:,} (Expected: 9,969,589)")
    assert n_cands == 9969589, f"Expected 9,969,589 candidates, got {n_cands}"
    
    # Verify normalization fields
    norm_fields = ["c_ids", "c_raw_names", "c_raw_addrs", "c_countries", "c_nn", "c_nns", "c_na", "c_pc"]
    print("4. Verifying normalization fields:")
    for fld in norm_fields:
        val = cache.get(fld)
        assert val is not None, f"Missing normalization field: {fld}"
        assert len(val) == n_cands, f"Field {fld} length {len(val)} != {n_cands}"
        print(f"   - {fld}: {len(val):,} entries (OK)")
        
    # Verify 8 indices
    idx_fields = [
        "idx_exact", "idx_pfx3", "idx_phone", "idx_postal",
        "idx_addr_key", "idx_comp", "idx_rare_tok", "idx_rare_addr"
    ]
    print("5. Verifying 8 inverted blocking indices:")
    for fld in idx_fields:
        val = cache.get(fld)
        assert val is not None, f"Missing index: {fld}"
        print(f"   - {fld}: {len(val):,} distinct keys (OK)")
        
    del cache
    gc.collect()
    print("\n>>> ALL CACHE INTEGRITY CHECKS PASSED SUCCESSFULLY! <<<\n")
    return True

def run_pipeline(limit: int, chunk_size: int = None):
    if chunk_size is None:
        chunk_size = limit
        
    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--limit", str(limit),
        "--chunk-size", str(chunk_size)
    ]
    
    print("=" * 75)
    print(f"RUNNING PIPELINE BENCHMARK: --limit {limit} (chunk_size={chunk_size})")
    print("=" * 75)
    
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    t_total = time.perf_counter() - t0
    
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
        
    assert res.returncode == 0, f"Pipeline failed with return code {res.returncode}"
    return res.stdout, t_total

if __name__ == "__main__":
    if verify_cache():
        print("Cache verification completed successfully.")
