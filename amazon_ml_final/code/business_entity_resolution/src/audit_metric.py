#!/usr/bin/env python3
"""
TASK 1: Audit Macro F0.5 Metric
Tests edge cases, singleton handling, multi-match, official challenge example,
and verifies mathematical exactness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import f05_per_entity, f05_macro

def run_tests():
    print("=== AUDITING MACRO F0.5 EVALUATOR ===")
    
    # 1. Official README Example:
    # S1-00001: true = [S2-00047, S3-00812], pred = [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0, F0.5 = 5/7 = 0.7142857
    t1 = {"S2-00047", "S3-00812"}
    p1 = {"S2-00047", "S2-00193", "S3-00812"}
    f, p, r = f05_per_entity(t1, p1)
    assert abs(f - 5/7) < 1e-5, f"Expected 0.714286, got {f}"
    print("  [PASS] Official README example gives exactly 0.714286")
    
    # 2. Perfect multi-match:
    t2 = {"S2-1", "S3-2"}
    p2 = {"S2-1", "S3-2"}
    f, p, r = f05_per_entity(t2, p2)
    assert abs(f - 1.0) < 1e-5, f"Expected 1.0, got {f}"
    print("  [PASS] Perfect multi-match gives 1.0")
    
    # 3. Singleton with correct empty prediction:
    t3 = set()
    p3 = set()
    f, p, r = f05_per_entity(t3, p3)
    assert f == 1.0, f"Expected 1.0 for singleton, got {f}"
    print("  [PASS] Singleton correctly predicted as empty gives 1.0")
    
    # 4. Singleton with false merge (FP):
    t4 = set()
    p4 = {"S2-99"}
    f, p, r = f05_per_entity(t4, p4)
    assert f == 0.0, f"Expected 0.0 for false merge singleton, got {f}"
    print("  [PASS] Singleton with false positive gives 0.0")
    
    # 5. Non-singleton with missed match (FN):
    t5 = {"S2-1"}
    p5 = set()
    f, p, r = f05_per_entity(t5, p5)
    assert f == 0.0, f"Expected 0.0 for missed match, got {f}"
    print("  [PASS] Non-singleton predicted as empty gives 0.0")
    
    # 6. Completely disjoint prediction (wrong IDs):
    t6 = {"S2-1"}
    p6 = {"S2-2"}
    f, p, r = f05_per_entity(t6, p6)
    assert f == 0.0, f"Expected 0.0 for disjoint prediction, got {f}"
    print("  [PASS] Disjoint prediction gives 0.0")
    
    # 7. Partial match (1 TP out of 2 true, 0 FP):
    # P = 1/1 = 1.0, R = 1/2 = 0.5 -> F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.833333
    t7 = {"S2-1", "S3-2"}
    p7 = {"S2-1"}
    f, p, r = f05_per_entity(t7, p7)
    assert abs(f - 5/6) < 1e-5, f"Expected 0.833333, got {f}"
    print("  [PASS] Partial match without false positives (P=1.0, R=0.5) gives 0.833333")
    
    # 8. Macro Averaging across 4 entities:
    # E1: 1.0, E2: 0.7142857, E3: 1.0 (singleton), E4: 0.0 (missed singleton)
    # Macro avg = (1.0 + 5/7 + 1.0 + 0.0) / 4 = 2.7142857 / 4 = 0.6785714
    truth = {
        "S1-1": t2,
        "S1-2": t1,
        "S1-3": t3,
        "S1-4": t4
    }
    preds = {
        "S1-1": p2,
        "S1-2": p1,
        "S1-3": p3,
        "S1-4": p4
    }
    macro_res = f05_macro(preds, truth)
    expected_macro = (1.0 + 5/7 + 1.0 + 0.0) / 4
    assert abs(macro_res["macro_f05"] - expected_macro) < 1e-5
    assert macro_res["singleton_acc"] == 0.5  # 1 out of 2 singletons correct
    print(f"  [PASS] Macro F0.5 across mixed batch: {macro_res['macro_f05']:.6f} == {expected_macro:.6f}")
    print(f"  [PASS] Singleton accuracy correctly isolated: {macro_res['singleton_acc']:.4f}")
    print("\nALL METRIC AUDIT CHECKS PASSED PERFECTLY.")

if __name__ == "__main__":
    run_tests()
