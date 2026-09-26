"""
Validation script to measure actual F0.5 score on held-out data

This addresses Issue #3: measure real F0.5, not assumed

Usage:
    python validate_pipeline.py --train_path <path> --split_ratio 0.8
"""
import pandas as pd
import sys
import os
from collections import defaultdict
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scorer import macro_f05
from normalization import normalize_business_name, normalize_address
from rapidfuzz import fuzz


def load_ground_truth(train_path):
    """
    Load ground truth matches from training data

    Returns:
        y_true: dict mapping S1 entity_id -> set of matched S2/S3 ids
        s1_df, s2_df, s3_df: DataFrames
    """
    print(f"Loading training data from {train_path}...")

    s1 = pd.read_csv(f'{train_path}/train_source1.tsv', sep='\t')
    s2 = pd.read_csv(f'{train_path}/train_source2.tsv', sep='\t')
    s3 = pd.read_csv(f'{train_path}/train_source3.tsv', sep='\t')
    ground_truth = pd.read_csv(f'{train_path}/train_ground_truth.tsv', sep='\t')

    print(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")
    print(f"  Ground truth rows: {len(ground_truth):,}")

    # Build ground truth dict from aggregated format
    # Format: source1_entity_id -> matched_entity_ids (comma-separated)
    y_true = {}

    for _, row in ground_truth.iterrows():
        s1_id = row['source1_entity_id']
        matched_ids = row['matched_entity_ids']

        if pd.isna(matched_ids) or matched_ids == '':
            y_true[s1_id] = set()
        else:
            # Split comma-separated IDs
            y_true[s1_id] = set(matched_ids.split(','))

    # Ensure all S1 entities have an entry (some may have no matches)
    for s1_id in s1['entity_id']:
        if s1_id not in y_true:
            y_true[s1_id] = set()

    # Count total matches
    total_matches = sum(len(v) for v in y_true.values())

    print(f"  Total ground truth matches: {total_matches:,}")
    print(f"  S1 entities with matches: {sum(1 for v in y_true.values() if v):,}")
    print(f"  S1 entities with NO matches: {sum(1 for v in y_true.values() if not v):,}")

    return y_true, s1, s2, s3


def train_val_split(y_true, s1_df, split_ratio=0.8, random_state=42):
    """
    Split S1 entities into train and validation sets

    Returns:
        train_s1_ids, val_s1_ids (sets)
    """
    import numpy as np

    np.random.seed(random_state)

    all_s1_ids = list(s1_df['entity_id'])
    np.random.shuffle(all_s1_ids)

    split_idx = int(len(all_s1_ids) * split_ratio)
    train_ids = set(all_s1_ids[:split_idx])
    val_ids = set(all_s1_ids[split_idx:])

    print(f"\nTrain/Val split ({split_ratio:.0%} train):")
    print(f"  Train: {len(train_ids):,} S1 entities")
    print(f"  Val: {len(val_ids):,} S1 entities")

    # Check match distribution
    train_with_matches = sum(1 for sid in train_ids if y_true[sid])
    val_with_matches = sum(1 for sid in val_ids if y_true[sid])

    print(f"  Train entities with matches: {train_with_matches:,}")
    print(f"  Val entities with matches: {val_with_matches:,}")

    return train_ids, val_ids


def run_simple_matcher(s1_df, s2_df, s3_df, s1_ids_filter=None,
                       name_thresh=85, addr_thresh=70,
                       max_candidates=50):
    """
    Run the simple rule-based matcher (mimics memory_efficient_submission.py OLD version)

    Returns:
        y_pred: dict mapping S1 entity_id -> set of predicted matches
    """
    print(f"\nRunning simple matcher (name>={name_thresh}, addr>={addr_thresh})...")

    # Normalize
    s1_df = s1_df.copy()
    s1_df['name_norm'] = s1_df['business_name'].apply(normalize_business_name)
    s1_df['addr_norm'] = s1_df['business_address'].apply(normalize_address)

    s2_df = s2_df.copy()
    s2_df['name_norm'] = s2_df['business_name'].apply(normalize_business_name)
    s2_df['addr_norm'] = s2_df['business_address'].apply(normalize_address)

    s3_df = s3_df.copy()
    s3_df['name_norm'] = s3_df['business_name'].apply(normalize_business_name)
    s3_df['addr_norm'] = s3_df['business_address'].apply(normalize_address)

    target = pd.concat([s2_df, s3_df], ignore_index=True)

    # Filter S1 if requested
    if s1_ids_filter:
        s1_df = s1_df[s1_df['entity_id'].isin(s1_ids_filter)]

    y_pred = {}

    print(f"  Processing {len(s1_df):,} S1 entities...")

    for idx, s1_row in s1_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"    {idx+1:,} / {len(s1_df):,}...")

        s1_id = s1_row['entity_id']

        # Simple blocking: name prefix (3 chars) + country
        name_prefix = s1_row['name_norm'][:3] if len(s1_row['name_norm']) >= 3 else s1_row['name_norm']

        candidates = target[
            (target['country'] == s1_row['country']) &
            (target['name_norm'].str.startswith(name_prefix))
        ]

        # OLD BUG: just take .head(50) without sorting
        # NEW: sort by quick similarity first
        if len(candidates) > max_candidates:
            # Quick score
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

    print(f"  Predicted {sum(len(v) for v in y_pred.values()):,} total matches")

    return y_pred


def resolve_conflicts(y_pred, s1_df, target_df):
    """
    Apply conflict resolution: if multiple S1 claim same target, keep highest confidence

    Returns:
        y_pred_resolved: dict with conflicts resolved
    """
    print("\nApplying conflict resolution...")

    # Normalize for similarity computation
    s1_df = s1_df.copy()
    s1_df['name_norm'] = s1_df['business_name'].apply(normalize_business_name)
    s1_df['addr_norm'] = s1_df['business_address'].apply(normalize_address)
    s1_indexed = s1_df.set_index('entity_id')

    target_df = target_df.copy()
    target_df['name_norm'] = target_df['business_name'].apply(normalize_business_name)
    target_df['addr_norm'] = target_df['business_address'].apply(normalize_address)
    target_indexed = target_df.set_index('entity_id')

    # Build claims map
    target_claims = defaultdict(list)

    for s1_id, match_set in y_pred.items():
        s1_row = s1_indexed.loc[s1_id]

        for target_id in match_set:
            t_row = target_indexed.loc[target_id]

            # Confidence = name_sim + addr_sim
            name_sim = fuzz.ratio(s1_row['name_norm'], t_row['name_norm'])
            addr_sim = fuzz.ratio(s1_row['addr_norm'], t_row['addr_norm'])
            confidence = name_sim + addr_sim

            target_claims[target_id].append((s1_id, confidence))

    # Find conflicts
    conflicts = {tid: claims for tid, claims in target_claims.items() if len(claims) > 1}
    print(f"  Found {len(conflicts):,} contested targets")

    if not conflicts:
        return y_pred

    # Resolve: highest confidence wins
    winners = {}
    for target_id, claims in conflicts.items():
        claims.sort(key=lambda x: x[1], reverse=True)
        winners[target_id] = claims[0][0]

    # Apply resolution
    y_pred_resolved = {}
    removed = 0

    for s1_id, match_set in y_pred.items():
        resolved_set = set()

        for target_id in match_set:
            if target_id in winners:
                if winners[target_id] == s1_id:
                    resolved_set.add(target_id)
                else:
                    removed += 1
            else:
                resolved_set.add(target_id)

        y_pred_resolved[s1_id] = resolved_set

    print(f"  Removed {removed:,} matches due to conflicts")

    return y_pred_resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../../../student_resource/dataset/train',
                       help='Path to training data directory')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--name_thresh', type=int, default=85,
                       help='Name similarity threshold (default: 85)')
    parser.add_argument('--addr_thresh', type=int, default=70,
                       help='Address similarity threshold (default: 70)')
    parser.add_argument('--max_candidates', type=int, default=50,
                       help='Max candidates per entity (default: 50)')
    parser.add_argument('--no_conflict_resolution', action='store_true',
                       help='Disable conflict resolution (to measure impact)')

    args = parser.parse_args()

    print("="*80)
    print("PIPELINE VALIDATION - Measure Actual F0.5")
    print("="*80)

    # Load data
    y_true, s1_df, s2_df, s3_df = load_ground_truth(args.train_path)

    # Split
    train_ids, val_ids = train_val_split(y_true, s1_df, args.split_ratio)

    # Get validation ground truth
    y_true_val = {sid: y_true[sid] for sid in val_ids}

    # Run matcher on validation set
    y_pred_val = run_simple_matcher(
        s1_df, s2_df, s3_df,
        s1_ids_filter=val_ids,
        name_thresh=args.name_thresh,
        addr_thresh=args.addr_thresh,
        max_candidates=args.max_candidates
    )

    # Conflict resolution
    if not args.no_conflict_resolution:
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)
        y_pred_val = resolve_conflicts(y_pred_val, s1_df, target_df)
    else:
        print("\nSkipping conflict resolution (--no_conflict_resolution)")

    # Score
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    score, details = macro_f05(y_true_val, y_pred_val, return_details=True)

    print(f"\nMacro F0.5 Score: {score:.4f}")

    # Aggregate stats
    precisions = [d['precision'] for d in details.values()]
    recalls = [d['recall'] for d in details.values()]

    import numpy as np
    print(f"\nPrecision: mean={np.mean(precisions):.3f}, median={np.median(precisions):.3f}")
    print(f"Recall: mean={np.mean(recalls):.3f}, median={np.median(recalls):.3f}")

    # Breakdown by match count
    with_matches = [sid for sid in y_true_val if y_true_val[sid]]
    without_matches = [sid for sid in y_true_val if not y_true_val[sid]]

    print(f"\nBreakdown:")
    print(f"  Entities WITH ground truth matches: {len(with_matches):,}")
    if with_matches:
        f05_with = [details[sid]['f05'] for sid in with_matches]
        print(f"    Mean F0.5: {np.mean(f05_with):.3f}")

    print(f"  Entities WITHOUT ground truth matches: {len(without_matches):,}")
    if without_matches:
        f05_without = [details[sid]['f05'] for sid in without_matches]
        print(f"    Mean F0.5: {np.mean(f05_without):.3f}")

    # Error analysis: worst performers
    print("\n" + "="*80)
    print("ERROR ANALYSIS - Worst 10 entities by F0.5")
    print("="*80)

    sorted_entities = sorted(details.items(), key=lambda x: x[1]['f05'])[:10]

    for s1_id, metrics in sorted_entities:
        true_set = y_true_val[s1_id]
        pred_set = y_pred_val.get(s1_id, set())

        print(f"\n{s1_id}:")
        print(f"  F0.5: {metrics['f05']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        print(f"  True matches: {len(true_set)}, Predicted: {len(pred_set)}")
        print(f"  True positives: {len(true_set & pred_set)}")
        if true_set - pred_set:
            print(f"  False negatives (missed): {true_set - pred_set}")
        if pred_set - true_set:
            print(f"  False positives (wrong): {pred_set - true_set}")

    print("\n" + "="*80)
    print("THRESHOLD SENSITIVITY")
    print("="*80)
    print("\nCurrent thresholds:")
    print(f"  Name >= {args.name_thresh}")
    print(f"  Address >= {args.addr_thresh}")
    print("\nTo test other thresholds, re-run with:")
    print("  python validate_pipeline.py --name_thresh 80 --addr_thresh 65")


if __name__ == '__main__':
    main()
