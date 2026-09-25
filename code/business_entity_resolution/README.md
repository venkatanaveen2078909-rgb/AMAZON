# Business Entity Resolution — High-Performance Competition Pipeline
**Macro-Averaged $F_{0.5}$ Champion Pipeline for ML Challenge 2026**

---

## 1. Quick Start & Execution

### Environment Setup
```bash
pip install -r requirements.txt
```

### Reproduce Validation Experiments & Metric Progression
```bash
# 1. Audit Metric & Unit Tests
python src/audit_metric.py

# 2. Experiment 01: Fast Baseline
python src/exp01_fast_baseline.py

# 3. Experiment 02: Multi-View Blocking Evaluation (98.66% candidate recall)
python src/exp02_blocking.py

# 4. Experiment 03: 48-Dimensional GBDT Pairwise Classifier Training
python src/exp03_pipeline.py

# 5. Experiment 04: Tiered Evidence Calibration & Contradiction Veto Filter
python src/exp04_rule_calibration.py
```

### Run Full Production Inference
Generates official `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python src/run_inference.py
```

### Validate Submission Output
```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

---

## 2. Architecture & Methodology

```
+──────────────────────────────────────────────────────────────────────────+
|                    END-TO-END PIPELINE ARCHITECTURE                      |
+──────────────────────────────────────────────────────────────────────────+
|  [Source 1]         [Source 2 & Source 3 Candidate Pool (9.97M rows)]    |
|      │                                   │                               |
|      ▼                                   ▼                               |
|  Country-Agnostic Normalization & Postal Code / Phonetic Extraction      |
|      │                                   │                               |
|      └───────────────────┬───────────────┘                               |
|                          ▼                                               |
|      8-Channel Multi-View Inverted Blocking (98.66% Recall)              |
|      - Exact Sorted Name    - Prefix-3          - Double Metaphone       |
|      - Rare Name Tokens     - Postal Code       - Addr Num + First Word  |
|      - Name2 + AddrNum      - Rare Landmark Address Tokens                |
|                          │                                               |
|                          ▼                                               |
|      48-Dimensional Pairwise Feature Vector Extraction                   |
|      (Names, Addresses, Phonetics, Geo-structure, Contradiction Flags)   |
|                          │                                               |
|                          ▼                                               |
|      LightGBM GBDT Binary Classifier (Hard-Negative Trained)             |
|                          │                                               |
|                          ▼                                               |
|      Tiered Evidence Calibration & Decision Engine                       |
|      - Contradiction Veto Filter (Postal mismatch + addr dissimilar)     |
|      - High Concordance Boost: Threshold = 0.85                          |
|      - Missing Address Protection: Threshold = 0.99                      |
|      - Base Evidence Threshold: Threshold = 0.97                         |
|      - Relative Margin Pruning: Δ_p ≤ 0.15                               |
|                          │                                               |
|                          ▼                                               |
|      Guaranteed Output: matched_ids ⊆ candidate_ids                      |
|      (output/matching_results.tsv & output/candidate_pairs.tsv)          |
+──────────────────────────────────────────────────────────────────────────+
```

---

## 3. Empirical Results & Validation Progression

| Experiment ID | Validation Macro-$F_{0.5}$ | Singleton Accuracy | Multi-Match $F_{0.5}$ | Precision | Recall | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-01 Baseline` | 0.8364 | 0.6421 | 0.8475 | 0.8920 | 0.9182 | 4,559 |
| `EXP-02C Blocking` | *(Recall = 98.66%)* | - | - | - | 0.9866 | - |
| `EXP-03 GBDT (48 Feat)` | 0.9394 | 0.8745 | 0.9432 | 0.9476 | 0.9589 | 1,167 |
| `EXP-04 Tiered Rules` | **0.9456** | **0.9077** | **0.9478** | **0.9668** | **0.9288** | **656** |

---

## 4. Key Engineering Innovations

1. **Missing Address Protection**: When address data is absent in candidate records, the classifier requires extreme certainty ($p \ge 0.99$) before merging, eliminating false merges on common corporate entity names.
2. **Contradiction Veto Filter**: Penalizes entity pairs with matching names but conflicting street addresses and postal codes within the same metropolitan area.
3. **8-Channel High-Recall Blocking**: Captures 98.66% of all true entity links across 9.97M candidates while reducing pairwise comparisons by >99.98%.
4. **UTF-8 & Country Invariance**: Zero hardcoded assumptions; fully open-set compatible (e.g. France records in test data).
