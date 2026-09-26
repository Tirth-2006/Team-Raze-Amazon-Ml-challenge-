"""
Optimized rule-based baseline - processes test set efficiently
Uses vectorized operations where possible
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def create_submission():
    """Create submission files using efficient rule-based matching"""

    print("="*80)
    print("EFFICIENT RULE-BASED BASELINE")
    print("="*80)

    # Load test data
    print("\nLoading test data...")
    base_path = '../../../student_resource/dataset/test'

    test_s1 = pd.read_csv(f'{base_path}/test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv(f'{base_path}/test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv(f'{base_path}/test_source3.tsv', sep='\t')

    print(f"Loaded: S1={len(test_s1):,}, S2={len(test_s2):,}, S3={len(test_s3):,}")

    # Normalize all names and addresses upfront
    print("\nNormalizing names and addresses...")
    test_s1['name_norm'] = test_s1['business_name'].apply(normalize_business_name)
    test_s1['addr_norm'] = test_s1['business_address'].apply(normalize_address)
    test_s1['name_first_3'] = test_s1['name_norm'].str[:3]

    test_s2['name_norm'] = test_s2['business_name'].apply(normalize_business_name)
    test_s2['addr_norm'] = test_s2['business_address'].apply(normalize_address)
    test_s2['name_first_3'] = test_s2['name_norm'].str[:3]

    test_s3['name_norm'] = test_s3['business_name'].apply(normalize_business_name)
    test_s3['addr_norm'] = test_s3['business_address'].apply(normalize_address)
    test_s3['name_first_3'] = test_s3['name_norm'].str[:3]

    # Combine targets
    target_df = pd.concat([test_s2, test_s3], ignore_index=True)

    print(f"Target pool: {len(target_df):,} records")

    # Process in batches
    batch_size = 500
    num_batches = (len(test_s1) + batch_size - 1) // batch_size

    all_candidates = {}
    all_matches = {}

    print(f"\nProcessing {len(test_s1):,} S1 entities in {num_batches} batches...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(test_s1))

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            print(f"Batch {batch_idx+1}/{num_batches}: Processing entities {start:,} to {end:,}...")

        s1_batch = test_s1.iloc[start:end]

        for _, s1_row in s1_batch.iterrows():
            s1_id = s1_row['entity_id']

            # Filter targets by country and name prefix
            target_filtered = target_df[
                (target_df['country'] == s1_row['country']) &
                (target_df['name_first_3'] == s1_row['name_first_3'])
            ]

            if len(target_filtered) == 0:
                all_candidates[s1_id] = set()
                all_matches[s1_id] = set()
                continue

            candidates_for_entity = set()
            matches_for_entity = set()

            # Check similarities
            for _, target_row in target_filtered.iterrows():
                target_id = target_row['entity_id']
                candidates_for_entity.add(target_id)

                # Compute similarities
                name_sim = fuzz.ratio(s1_row['name_norm'], target_row['name_norm'])
                addr_sim = fuzz.ratio(s1_row['addr_norm'], target_row['addr_norm'])

                # Match if both name and address are similar
                if name_sim >= 85 and addr_sim >= 75:
                    matches_for_entity.add(target_id)

            all_candidates[s1_id] = candidates_for_entity
            all_matches[s1_id] = matches_for_entity

    # Write output files
    print("\n" + "="*80)
    print("WRITING OUTPUT FILES")
    print("="*80)

    output_dir = '../../../output'
    os.makedirs(output_dir, exist_ok=True)

    # candidate_pairs.tsv
    candidate_file = f'{output_dir}/candidate_pairs.tsv'
    print(f"Writing {candidate_file}...")
    with open(candidate_file, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1['entity_id']:
            cands = all_candidates.get(s1_id, set())
            cand_str = ','.join(sorted(cands)) if cands else ''
            f.write(f"{s1_id}\t{cand_str}\n")

    # matching_results.tsv
    matching_file = f'{output_dir}/matching_results.tsv'
    print(f"Writing {matching_file}...")
    with open(matching_file, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1['entity_id']:
            matches = all_matches.get(s1_id, set())
            match_str = ','.join(sorted(matches)) if matches else ''
            f.write(f"{s1_id}\t{match_str}\n")

    # Summary statistics
    total_cands = sum(len(v) for v in all_candidates.values())
    total_matches = sum(len(v) for v in all_matches.values())
    entities_with_matches = sum(1 for v in all_matches.values() if v)
    singletons = len(all_matches) - entities_with_matches

    print(f"\n=== SUMMARY ===")
    print(f"Total S1 entities: {len(all_matches):,}")
    print(f"Entities with matches: {entities_with_matches:,}")
    print(f"Singletons: {singletons:,}")
    print(f"Total candidates: {total_cands:,}")
    print(f"Total predicted matches: {total_matches:,}")
    print(f"Avg candidates per entity: {total_cands/len(all_candidates):.1f}")
    print(f"Avg matches per entity: {total_matches/len(all_matches):.2f}")

    print("\n" + "="*80)
    print("COMPLETE - Files written to output/")
    print("="*80)
    print("\nNext: Run validation script to check format")
    print("  python ../../../student_resource/utils/validate_submission.py \\")
    print("    --matching ../../../output/matching_results.tsv \\")
    print("    --candidate ../../../output/candidate_pairs.tsv \\")
    print("    --test-dir ../../../student_resource/dataset/test")


if __name__ == '__main__':
    create_submission()
