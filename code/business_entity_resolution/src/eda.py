#!/usr/bin/env python3
"""Quick EDA on training data — sizes, country distribution, match stats."""

import pandas as pd
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "6ab10eb3b23ba_student_resource", "student_resource", "dataset")

def main():
    # --- Training data ---
    print("=== TRAINING DATA ===")
    for name in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(BASE, "train", name)
        # Count lines without loading full file
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1  # subtract header
        print(f"{name}: {n:,} rows")

    # Load ground truth (smaller)
    gt = pd.read_csv(os.path.join(BASE, "train", "train_ground_truth.tsv"), sep="\t")
    print(f"\nGround truth rows: {len(gt):,}")
    
    # Count singletons (empty matched_entity_ids)
    singletons = gt["matched_entity_ids"].isna().sum()
    print(f"Singletons (no match): {singletons:,} ({100*singletons/len(gt):.1f}%)")
    
    # Count multi-match
    non_singleton = gt[gt["matched_entity_ids"].notna()]
    match_counts = non_singleton["matched_entity_ids"].str.split(",").apply(len)
    print(f"Non-singletons: {len(non_singleton):,}")
    print(f"  Mean matches per non-singleton: {match_counts.mean():.2f}")
    print(f"  Max matches: {match_counts.max()}")
    print(f"  Distribution:\n{match_counts.value_counts().sort_index().head(15)}")
    
    # Count S2 vs S3 matches
    all_matches = non_singleton["matched_entity_ids"].str.split(",").explode()
    s2_count = all_matches.str.startswith("S2-").sum()
    s3_count = all_matches.str.startswith("S3-").sum()
    print(f"\nTotal matched IDs: {len(all_matches):,}")
    print(f"  S2 matches: {s2_count:,}")
    print(f"  S3 matches: {s3_count:,}")
    
    # --- Test data ---
    print("\n=== TEST DATA ===")
    for name in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(BASE, "test", name)
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print(f"{name}: {n:,} rows")
    
    # Country distribution for train S1 (sample first 10k to be fast)
    print("\n=== COUNTRY DISTRIBUTION (train S1, first 50k) ===")
    s1_sample = pd.read_csv(os.path.join(BASE, "train", "train_source1.tsv"), 
                             sep="\t", nrows=50000)
    print(s1_sample["country"].value_counts())
    
    print("\n=== COUNTRY DISTRIBUTION (test S1, first 50k) ===")
    s1_test_sample = pd.read_csv(os.path.join(BASE, "test", "test_source1.tsv"), 
                                  sep="\t", nrows=50000)
    print(s1_test_sample["country"].value_counts())
    
    # Sample some actual records to understand patterns
    print("\n=== SAMPLE RECORDS (train S1, US) ===")
    us_sample = s1_sample[s1_sample["country"] == "US"].head(5)
    for _, r in us_sample.iterrows():
        print(f"  {r['entity_id']}: {r['business_name']} | {r['business_address']}")
    
    print("\n=== SAMPLE RECORDS (train S1, India) ===")
    in_sample = s1_sample[s1_sample["country"] == "India"].head(5)
    for _, r in in_sample.iterrows():
        print(f"  {r['entity_id']}: {r['business_name']} | {r['business_address']}")
    
    print("\n=== SAMPLE RECORDS (test S1, France) ===")
    fr_sample = s1_test_sample[s1_test_sample["country"] == "France"].head(5)
    for _, r in fr_sample.iterrows():
        print(f"  {r['entity_id']}: {r['business_name']} | {r['business_address']}")
    
    # Check for missing values
    print("\n=== MISSING VALUES (train S1, first 50k) ===")
    print(s1_sample.isnull().sum())
    print(f"\nEmpty business_name: {(s1_sample['business_name'] == '').sum()}")
    print(f"Empty business_address: {(s1_sample['business_address'] == '').sum()}")

if __name__ == "__main__":
    main()
