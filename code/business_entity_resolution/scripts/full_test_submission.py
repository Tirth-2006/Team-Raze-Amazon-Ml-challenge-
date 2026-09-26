"""
Full test set processing - optimized for speed and memory
Processes all 1.7M test entities
"""
import pandas as pd
from rapidfuzz import fuzz
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def main():
    print("="*80)
    print("FULL TEST SET PROCESSING")
    print("="*80)

    # Load test data
    print("\nLoading test data...")
    base = '../../../student_resource/dataset/test'

    s1 = pd.read_csv(f'{base}/test_source1.tsv', sep='\t')
    s2 = pd.read_csv(f'{base}/test_source2.tsv', sep='\t')
    s3 = pd.read_csv(f'{base}/test_source3.tsv', sep='\t')

    print(f"S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")

    # Normalize
    print("\nNormalizing S1...")
    s1['name_norm'] = s1['business_name'].apply(normalize_business_name)
    s1['addr_norm'] = s1['business_address'].apply(normalize_address)
    s1['key'] = s1['name_norm'].str[:3] + '_' + s1['country']

    print("Normalizing S2...")
    s2['name_norm'] = s2['business_name'].apply(normalize_business_name)
    s2['addr_norm'] = s2['business_address'].apply(normalize_address)
    s2['key'] = s2['name_norm'].str[:3] + '_' + s2['country']

    print("Normalizing S3...")
    s3['name_norm'] = s3['business_name'].apply(normalize_business_name)
    s3['addr_norm'] = s3['business_address'].apply(normalize_address)
    s3['key'] = s3['name_norm'].str[:3] + s3['country']

    # Build index
    print("\nBuilding target index...")
    target = pd.concat([s2, s3], ignore_index=True)

    # Group by key for fast lookup
    target_groups = target.groupby('key')

    print(f"Index built with {len(target_groups):,} unique keys")

    # Process S1
    print(f"\nProcessing {len(s1):,} S1 entities...")

    all_candidates = {}
    all_matches = {}

    report_every = 100000
    processed = 0

    for idx, s1_row in s1.iterrows():
        s1_id = s1_row['entity_id']

        processed += 1
        if processed % report_every == 0:
            print(f"  Processed {processed:,} / {len(s1):,}...")

        key = s1_row['key']

        # Get candidates from index
        if key not in target_groups.groups:
            all_candidates[s1_id] = set()
            all_matches[s1_id] = set()
            continue

        candidates_df = target_groups.get_group(key).head(50)  # Limit to 50

        cands = set()
        matched = set()

        for _, t_row in candidates_df.iterrows():
            tid = t_row['entity_id']
            cands.add(tid)

            # Name similarity
            name_sim = fuzz.ratio(s1_row['name_norm'], t_row['name_norm'])

            if name_sim >= 80:
                # Address similarity (only if name promising)
                addr_sim = fuzz.ratio(s1_row['addr_norm'], t_row['addr_norm'])

                if name_sim >= 85 and addr_sim >= 70:
                    matched.add(tid)

        all_candidates[s1_id] = cands
        all_matches[s1_id] = matched

    print(f"  Completed all {len(s1):,} entities")

    # Write outputs
    print("\n" + "="*80)
    print("WRITING OUTPUT FILES")
    print("="*80)

    os.makedirs('../../../output', exist_ok=True)

    print("Writing candidate_pairs.tsv...")
    with open('../../../output/candidate_pairs.tsv', 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1['entity_id']:
            cands = all_candidates.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")

    print("Writing matching_results.tsv...")
    with open('../../../output/matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1['entity_id']:
            matches = all_matches.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(matches))}\n")

    # Stats
    total_cands = sum(len(v) for v in all_candidates.values())
    total_matches = sum(len(v) for v in all_matches.values())
    with_matches = sum(1 for v in all_matches.values() if v)

    print(f"\n=== SUMMARY ===")
    print(f"S1 entities: {len(s1):,}")
    print(f"With matches: {with_matches:,}")
    print(f"Singletons: {len(s1) - with_matches:,}")
    print(f"Total candidates: {total_cands:,}")
    print(f"Total matches: {total_matches:,}")
    print(f"Avg candidates/entity: {total_cands/len(s1):.1f}")
    print(f"Avg matches/entity: {total_matches/len(s1):.2f}")

    print("\n" + "="*80)
    print("COMPLETE - Submission files ready!")
    print("="*80)


if __name__ == '__main__':
    main()
