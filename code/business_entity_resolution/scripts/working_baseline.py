"""
Working baseline - tested and optimized
Generates valid submission files in reasonable time
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def main():
    print("="*80)
    print("WORKING BASELINE SUBMISSION")
    print("="*80)

    # Load test data
    print("\nLoading test data...")
    base = '../../../student_resource/dataset/test'

    s1 = pd.read_csv(f'{base}/test_source1.tsv', sep='\t')
    s2 = pd.read_csv(f'{base}/test_source2.tsv', sep='\t')
    s3 = pd.read_csv(f'{base}/test_source3.tsv', sep='\t')

    print(f"Loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")

    # Normalize all data upfront
    print("\nNormalizing data...")

    print("  Normalizing S1...")
    s1['name_norm'] = s1['business_name'].apply(normalize_business_name)
    s1['addr_norm'] = s1['business_address'].apply(normalize_address)
    s1['name_first4'] = s1['name_norm'].str[:4]

    print("  Normalizing S2...")
    s2['name_norm'] = s2['business_name'].apply(normalize_business_name)
    s2['addr_norm'] = s2['business_address'].apply(normalize_address)
    s2['name_first4'] = s2['name_norm'].str[:4]

    print("  Normalizing S3...")
    s3['name_norm'] = s3['business_name'].apply(normalize_business_name)
    s3['addr_norm'] = s3['business_address'].apply(normalize_address)
    s3['name_first4'] = s3['name_norm'].str[:4]

    # Combine S2 and S3
    target = pd.concat([s2, s3], ignore_index=True)
    print(f"Target pool: {len(target):,} records")

    # Build lookup index
    print("\nBuilding index by (name_first4, country)...")
    from collections import defaultdict
    index = defaultdict(list)

    for idx, row in target.iterrows():
        key = (row['name_first4'], row['country'])
        index[key].append(idx)

    print(f"Index has {len(index):,} unique keys")

    # Process S1 entities
    print(f"\nProcessing {len(s1):,} S1 entities...")

    all_candidates = {}
    all_matches = {}

    processed = 0
    batch_report_interval = 10000

    for s1_idx, s1_row in s1.iterrows():
        s1_id = s1_row['entity_id']

        # Report progress
        processed += 1
        if processed % batch_report_interval == 0:
            print(f"  Processed {processed:,} / {len(s1):,} entities...")

        # Find candidates using index
        key = (s1_row['name_first4'], s1_row['country'])
        candidate_indices = index.get(key, [])

        if not candidate_indices:
            all_candidates[s1_id] = set()
            all_matches[s1_id] = set()
            continue

        # Limit candidates to first 100 for speed
        candidate_indices = candidate_indices[:100]

        candidates_for_this_entity = set()
        matches_for_this_entity = set()

        # Compare with each candidate
        for target_idx in candidate_indices:
            target_row = target.iloc[target_idx]
            target_id = target_row['entity_id']

            candidates_for_this_entity.add(target_id)

            # Compute similarities
            name_sim = fuzz.ratio(s1_row['name_norm'], target_row['name_norm'])

            # Only compute address if name is promising
            if name_sim >= 75:
                addr_sim = fuzz.ratio(s1_row['addr_norm'], target_row['addr_norm'])

                # Match if both similarities are high
                if name_sim >= 85 and addr_sim >= 70:
                    matches_for_this_entity.add(target_id)

        all_candidates[s1_id] = candidates_for_this_entity
        all_matches[s1_id] = matches_for_this_entity

    print(f"  Processed all {len(s1):,} entities")

    # Write output files
    print("\n" + "="*80)
    print("WRITING OUTPUT FILES")
    print("="*80)

    output_dir = '../../../output'
    os.makedirs(output_dir, exist_ok=True)

    # candidate_pairs.tsv
    candidate_file = f'{output_dir}/candidate_pairs.tsv'
    print(f"Writing {candidate_file}...")

    with open(candidate_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1['entity_id']:
            cands = all_candidates.get(s1_id, set())
            cand_str = ','.join(sorted(cands)) if cands else ''
            f.write(f"{s1_id}\t{cand_str}\n")

    # matching_results.tsv
    matching_file = f'{output_dir}/matching_results.tsv'
    print(f"Writing {matching_file}...")

    with open(matching_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1['entity_id']:
            matches = all_matches.get(s1_id, set())
            match_str = ','.join(sorted(matches)) if matches else ''
            f.write(f"{s1_id}\t{match_str}\n")

    # Summary statistics
    total_cands = sum(len(v) for v in all_candidates.values())
    total_matches = sum(len(v) for v in all_matches.values())
    entities_with_matches = sum(1 for v in all_matches.values() if v)
    singletons = len(all_matches) - entities_with_matches

    print(f"\n=== SUMMARY ===")
    print(f"Total S1 entities: {len(s1):,}")
    print(f"Entities with matches: {entities_with_matches:,}")
    print(f"Singletons (no match): {singletons:,}")
    print(f"Total candidate pairs: {total_cands:,}")
    print(f"Total predicted matches: {total_matches:,}")
    print(f"Avg candidates per entity: {total_cands/len(s1):.1f}")
    print(f"Avg matches per entity: {total_matches/len(s1):.2f}")

    print("\n" + "="*80)
    print("SUBMISSION FILES CREATED SUCCESSFULLY")
    print("="*80)
    print("\nNext step: Validate submission")
    print("  cd ../../../student_resource")
    print("  python utils/validate_submission.py \\")
    print("    --matching ../output/matching_results.tsv \\")
    print("    --candidate ../output/candidate_pairs.tsv \\")
    print("    --test-dir dataset/test")


if __name__ == '__main__':
    main()
