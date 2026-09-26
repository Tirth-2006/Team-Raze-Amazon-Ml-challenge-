"""
Quick validation test on small sample
Tests the pipeline logic without processing millions of records
"""
import pandas as pd
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scorer import macro_f05
from normalization import normalize_business_name, normalize_address
from rapidfuzz import fuzz


def quick_test(sample_size=1000):
    """Test on small sample to verify pipeline works"""

    print("="*80)
    print(f"QUICK VALIDATION TEST (sample size: {sample_size:,})")
    print("="*80)

    base = '../../../student_resource/dataset/train'

    # Load just the first N rows
    print(f"\nLoading sample of {sample_size:,} S1 entities...")
    s1 = pd.read_csv(f'{base}/train_source1.tsv', sep='\t', nrows=sample_size)
    s2 = pd.read_csv(f'{base}/train_source2.tsv', sep='\t', nrows=50000)  # Small lookup set
    s3 = pd.read_csv(f'{base}/train_source3.tsv', sep='\t', nrows=50000)

    ground_truth = pd.read_csv(f'{base}/train_ground_truth.tsv', sep='\t', nrows=sample_size)

    print(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")

    # Build ground truth dict
    y_true = {}
    for _, row in ground_truth.iterrows():
        s1_id = row['source1_entity_id']
        matched_ids = row['matched_entity_ids']

        if pd.isna(matched_ids) or matched_ids == '':
            y_true[s1_id] = set()
        else:
            y_true[s1_id] = set(matched_ids.split(','))

    # Ensure all S1 entities have entry
    for s1_id in s1['entity_id']:
        if s1_id not in y_true:
            y_true[s1_id] = set()

    total_gt = sum(len(v) for v in y_true.values())
    print(f"  Ground truth matches: {total_gt:,}")

    # Normalize
    print("\nNormalizing data...")
    s1['name_norm'] = s1['business_name'].apply(normalize_business_name)
    s1['addr_norm'] = s1['business_address'].apply(normalize_address)

    s2['name_norm'] = s2['business_name'].apply(normalize_business_name)
    s2['addr_norm'] = s2['business_address'].apply(normalize_address)

    s3['name_norm'] = s3['business_name'].apply(normalize_business_name)
    s3['addr_norm'] = s3['business_address'].apply(normalize_address)

    target = pd.concat([s2, s3], ignore_index=True)

    # Run simple matcher
    print("\nRunning simple matcher...")
    name_thresh = 85
    addr_thresh = 70
    max_candidates = 50

    y_pred = {}

    for idx, s1_row in s1.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1} / {len(s1)}...")

        s1_id = s1_row['entity_id']

        # Simple blocking: name prefix (3 chars) + country
        name_prefix = s1_row['name_norm'][:3] if len(s1_row['name_norm']) >= 3 else s1_row['name_norm']

        candidates = target[
            (target['country'] == s1_row['country']) &
            (target['name_norm'].str.startswith(name_prefix))
        ]

        # FIXED: Sort by similarity before taking top 50
        if len(candidates) > max_candidates:
            candidates = candidates.copy()
            candidates['quick_score'] = candidates.apply(
                lambda row: 0.6 * (fuzz.ratio(s1_row['name_norm'], row['name_norm']) / 100.0) +
                           0.4 * (fuzz.ratio(s1_row['addr_norm'], row['addr_norm']) / 100.0),
                axis=1
            )
            candidates = candidates.nlargest(max_candidates, 'quick_score')

        matches = set()

        for _, t_row in candidates.iterrows():
            name_sim = fuzz.ratio(s1_row['name_norm'], t_row['name_norm'])

            if name_sim >= name_thresh:
                addr_sim = fuzz.ratio(s1_row['addr_norm'], t_row['addr_norm'])
                if addr_sim >= addr_thresh:
                    matches.add(t_row['entity_id'])

        y_pred[s1_id] = matches

    total_pred = sum(len(v) for v in y_pred.values())
    print(f"\n  Predicted {total_pred:,} total matches")

    # Score
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    score, details = macro_f05(y_true, y_pred, return_details=True)

    print(f"\nMacro F0.5 Score: {score:.4f}")

    # Stats
    import numpy as np
    precisions = [d['precision'] for d in details.values()]
    recalls = [d['recall'] for d in details.values()]

    print(f"\nPrecision: mean={np.mean(precisions):.3f}, median={np.median(precisions):.3f}")
    print(f"Recall: mean={np.mean(recalls):.3f}, median={np.median(recalls):.3f}")

    # Sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 10 entities)")
    print("="*80)

    for i, (s1_id, metrics) in enumerate(list(details.items())[:10]):
        true_set = y_true[s1_id]
        pred_set = y_pred.get(s1_id, set())

        print(f"\n{s1_id}:")
        print(f"  F0.5: {metrics['f05']:.3f}, P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}")
        print(f"  True: {len(true_set)}, Pred: {len(pred_set)}, TP: {len(true_set & pred_set)}")

    print("\n" + "="*80)
    print("Test complete! Pipeline logic is working.")
    print("For full validation, run: python validate_pipeline.py")
    print("Warning: Full validation will take 30-60+ minutes")
    print("="*80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of S1 entities to test (default: 1000)')
    args = parser.parse_args()

    quick_test(args.sample_size)
