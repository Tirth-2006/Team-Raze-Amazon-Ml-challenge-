"""
Rule-based baseline - get a valid submission quickly
Uses simple similarity thresholds without ML training
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def simple_match(s1_row, target_row, name_threshold=80, addr_threshold=70):
    """
    Simple rule: match if name similarity > threshold AND address similarity > threshold
    AND country matches
    """
    if s1_row['country'] != target_row['country']:
        return False

    s1_name = normalize_business_name(s1_row['business_name'])
    target_name = normalize_business_name(target_row['business_name'])

    s1_addr = normalize_address(s1_row['business_address'])
    target_addr = normalize_address(target_row['business_address'])

    name_sim = fuzz.ratio(s1_name, target_name)
    addr_sim = fuzz.ratio(s1_addr, target_addr)

    return name_sim >= name_threshold and addr_sim >= addr_threshold


def process_batch(s1_batch, s2_df, s3_df, name_thresh=80, addr_thresh=70):
    """Process a batch of S1 entities"""
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    candidates = {}
    matches = {}

    for _, s1_row in s1_batch.iterrows():
        s1_id = s1_row['entity_id']

        # First pass: filter by country and first letter of name
        s1_name_norm = normalize_business_name(s1_row['business_name'])
        if len(s1_name_norm) == 0:
            candidates[s1_id] = set()
            matches[s1_id] = set()
            continue

        first_char = s1_name_norm[0]

        # Filter targets
        target_filtered = target_df[
            (target_df['country'] == s1_row['country'])
        ]

        # Further filter by name first character
        target_filtered = target_filtered[
            target_filtered['business_name'].apply(
                lambda x: normalize_business_name(x).startswith(first_char) if normalize_business_name(x) else False
            )
        ]

        # Collect candidates and matches
        cands = set()
        matched = set()

        for _, target_row in target_filtered.iterrows():
            target_id = target_row['entity_id']
            cands.add(target_id)

            if simple_match(s1_row, target_row, name_thresh, addr_thresh):
                matched.add(target_id)

        candidates[s1_id] = cands
        matches[s1_id] = matched

    return candidates, matches


def main():
    print("="*80)
    print("RULE-BASED BASELINE")
    print("="*80)

    # Load test data
    print("\nLoading test data...")
    test_s1 = pd.read_csv('../../../student_resource/dataset/test/test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv('../../../student_resource/dataset/test/test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv('../../../student_resource/dataset/test/test_source3.tsv', sep='\t')

    print(f"S1={len(test_s1):,}, S2={len(test_s2):,}, S3={len(test_s3):,}")

    # Process in batches
    batch_size = 1000
    all_candidates = {}
    all_matches = {}

    num_batches = (len(test_s1) + batch_size - 1) // batch_size

    print(f"\nProcessing {len(test_s1):,} entities in {num_batches} batches...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(test_s1))

        s1_batch = test_s1.iloc[start:end]

        print(f"\nBatch {batch_idx+1}/{num_batches}: entities {start:,}-{end:,}")

        cands, matches = process_batch(s1_batch, test_s2, test_s3, name_thresh=80, addr_thresh=70)

        all_candidates.update(cands)
        all_matches.update(matches)

        # Print progress
        batch_cands = sum(len(v) for v in cands.values())
        batch_matches = sum(len(v) for v in matches.values())
        print(f"  Candidates: {batch_cands:,}, Matches: {batch_matches:,}")

    # Write output
    print("\n" + "="*80)
    print("WRITING OUTPUT FILES")
    print("="*80)

    os.makedirs('../../../output', exist_ok=True)

    # candidate_pairs.tsv
    with open('../../../output/candidate_pairs.tsv', 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(all_candidates.keys()):
            cands = ','.join(sorted(all_candidates[s1_id])) if all_candidates[s1_id] else ''
            f.write(f"{s1_id}\t{cands}\n")

    # matching_results.tsv
    with open('../../../output/matching_results.tsv', 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(all_matches.keys()):
            matches = ','.join(sorted(all_matches[s1_id])) if all_matches[s1_id] else ''
            f.write(f"{s1_id}\t{matches}\n")

    # Stats
    total_cands = sum(len(v) for v in all_candidates.values())
    total_matches = sum(len(v) for v in all_matches.values())
    entities_with_matches = sum(1 for v in all_matches.values() if v)

    print(f"\n=== SUMMARY ===")
    print(f"Total S1 entities: {len(all_matches):,}")
    print(f"Entities with matches: {entities_with_matches:,}")
    print(f"Singletons: {len(all_matches) - entities_with_matches:,}")
    print(f"Total candidates: {total_cands:,}")
    print(f"Total matches: {total_matches:,}")
    print(f"Avg candidates/entity: {total_cands/len(all_candidates):.1f}")
    print(f"Avg matches/entity: {total_matches/len(all_matches):.1f}")

    print("\n" + "="*80)
    print("COMPLETE - Output files in ../../../output/")
    print("="*80)


if __name__ == '__main__':
    main()
