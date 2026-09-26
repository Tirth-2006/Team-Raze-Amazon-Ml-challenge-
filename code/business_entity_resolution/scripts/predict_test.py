"""
Test set inference script - generates submission files
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blocking import generate_candidates_multi_pass
from features import build_feature_matrix


def load_test_data(base_path='../../../student_resource/dataset'):
    """Load test data"""
    print("Loading test data...")
    test_s1 = pd.read_csv(f'{base_path}/test/test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv(f'{base_path}/test/test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv(f'{base_path}/test/test_source3.tsv', sep='\t')
    print(f"Loaded: S1={len(test_s1):,}, S2={len(test_s2):,}, S3={len(test_s3):,}")
    return test_s1, test_s2, test_s3


def load_model(model_path='../../../models/xgb_baseline.pkl'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_cols']


def load_threshold(threshold_path='../../../models/threshold.txt'):
    """Load threshold"""
    print(f"Loading threshold from {threshold_path}...")
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    print(f"Threshold: {threshold}")
    return threshold


def predict_test_set(
    test_s1_df,
    test_s2_df,
    test_s3_df,
    model,
    feature_cols,
    threshold,
    batch_size=10000
):
    """
    Run inference on test set in batches

    Returns:
        candidates_dict: {s1_id: set of candidate ids}
        predictions_dict: {s1_id: set of predicted match ids}
    """
    print("\n" + "="*80)
    print("BLOCKING - TEST SET")
    print("="*80)

    # Generate candidates
    candidates = generate_candidates_multi_pass(test_s1_df, test_s2_df, test_s3_df, verbose=True)

    print("\n" + "="*80)
    print("FEATURE ENGINEERING & PREDICTION - TEST SET")
    print("="*80)

    target_df = pd.concat([test_s2_df, test_s3_df], ignore_index=True)
    predictions = {}

    # Process in batches
    s1_ids = list(candidates.keys())
    num_batches = (len(s1_ids) + batch_size - 1) // batch_size

    print(f"Processing {len(s1_ids):,} S1 entities in {num_batches} batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(s1_ids))
        batch_s1_ids = s1_ids[start_idx:end_idx]

        print(f"\nBatch {batch_idx+1}/{num_batches}: entities {start_idx:,} to {end_idx:,}")

        # Collect candidate pairs for this batch
        batch_pairs = []
        for s1_id in batch_s1_ids:
            for target_id in candidates[s1_id]:
                batch_pairs.append((s1_id, target_id))

        if not batch_pairs:
            print("  No candidates in this batch")
            for s1_id in batch_s1_ids:
                predictions[s1_id] = set()
            continue

        print(f"  Building features for {len(batch_pairs):,} pairs...")

        # Build features
        batch_s1_df = test_s1_df[test_s1_df['entity_id'].isin(batch_s1_ids)]
        feature_df = build_feature_matrix(batch_s1_df, target_df, batch_pairs, verbose=False)

        # Predict
        X = feature_df[feature_cols].values
        probs = model.predict_proba(X)[:, 1]
        feature_df['prob'] = probs

        # Group by S1 entity and apply threshold
        for s1_id in batch_s1_ids:
            entity_preds = feature_df[feature_df['s1_id'] == s1_id]
            matched_ids = set(entity_preds[entity_preds['prob'] >= threshold]['target_id'])
            predictions[s1_id] = matched_ids

        print(f"  Predicted matches for {len(batch_s1_ids):,} entities")

    return candidates, predictions


def write_submission_files(
    candidates_dict,
    predictions_dict,
    output_dir='../../../output'
):
    """Write submission files"""
    print("\n" + "="*80)
    print("WRITING SUBMISSION FILES")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Write candidate_pairs.tsv
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    print(f"Writing {candidate_path}...")

    with open(candidate_path, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates_dict.keys()):
            cands = candidates_dict[s1_id]
            cand_str = ','.join(sorted(cands)) if cands else ''
            f.write(f"{s1_id}\t{cand_str}\n")

    # Write matching_results.tsv
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    print(f"Writing {matching_path}...")

    with open(matching_path, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(predictions_dict.keys()):
            matches = predictions_dict[s1_id]
            match_str = ','.join(sorted(matches)) if matches else ''
            f.write(f"{s1_id}\t{match_str}\n")

    print(f"\nSubmission files written to {output_dir}")

    # Summary stats
    total_candidates = sum(len(v) for v in candidates_dict.values())
    total_matches = sum(len(v) for v in predictions_dict.values())
    entities_with_matches = sum(1 for v in predictions_dict.values() if len(v) > 0)
    entities_singletons = sum(1 for v in predictions_dict.values() if len(v) == 0)

    print(f"\n=== SUMMARY ===")
    print(f"Total S1 entities: {len(predictions_dict):,}")
    print(f"Entities with matches: {entities_with_matches:,}")
    print(f"Singletons: {entities_singletons:,}")
    print(f"Total candidates: {total_candidates:,}")
    print(f"Total predicted matches: {total_matches:,}")
    print(f"Avg candidates per entity: {total_candidates/len(candidates_dict):.1f}")
    print(f"Avg matches per entity: {total_matches/len(predictions_dict):.1f}")


def main():
    print("="*80)
    print("TEST SET INFERENCE")
    print("="*80)

    # Load test data
    test_s1, test_s2, test_s3 = load_test_data()

    # Load model
    model, feature_cols = load_model()
    threshold = load_threshold()

    # Predict
    candidates, predictions = predict_test_set(
        test_s1, test_s2, test_s3,
        model, feature_cols, threshold,
        batch_size=10000
    )

    # Write outputs
    write_submission_files(candidates, predictions)

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
