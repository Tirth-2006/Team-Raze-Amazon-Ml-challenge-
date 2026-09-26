"""
Ultra-fast baseline - aggressive filtering to reduce comparisons
Focus: Get a valid submission quickly, then iterate
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def main():
    print("="*80)
    print("ULTRA-FAST BASELINE SUBMISSION")
    print("="*80)

    # Load data
    print("\nLoading test data...")
    base = '../../../student_resource/dataset/test'
    s1 = pd.read_csv(f'{base}/test_source1.tsv', sep='\t')
    s2 = pd.read_csv(f'{base}/test_source2.tsv', sep='\t')
    s3 = pd.read_csv(f'{base}/test_source3.tsv', sep='\t')

    print(f"S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")

    # Normalize
    print("\nNormalizing...")
    for df in [s1, s2, s3]:
        df['name_norm'] = df['business_name'].apply(normalize_business_name)
        df['addr_norm'] = df['business_address'].apply(normalize_address)
        df['name_prefix'] = df['name_norm'].str[:4]
        df['addr_num'] = df['addr_norm'].str.extract(r'(\d+)')[0].fillna('')

    # Combine targets
    s2['source'] = 'S2'
    s3['source'] = 'S3'
    target = pd.concat([s2, s3], ignore_index=True)

    # Build index: (name_prefix, country) -> list of targets
    print("\nBuilding index...")
    index = defaultdict(list)
    for idx, row in target.iterrows():
        key = (row['name_prefix'], row['country'])
        index[key].append({
            'id': row['entity_id'],
            'name': row['name_norm'],
            'addr': row['addr_norm'],
            'addr_num': row['addr_num']
        })

    print(f"Index built with {len(index):,} keys")

    # Process S1 entities
    print(f"\nProcessing {len(s1):,} S1 entities...")
    candidates = {}
    matches = {}

    batch_size = 10000
    for batch_start in range(0, len(s1), batch_size):
        batch_end = min(batch_start + batch_size, len(s1))

        if (batch_start // batch_size + 1) % 10 == 0 or batch_start == 0:
            print(f"  Batch {batch_start:,} to {batch_end:,}...")

        for idx in range(batch_start, batch_end):
            s1_row = s1.iloc[idx]
            s1_id = s1_row['entity_id']

            key = (s1_row['name_prefix'], s1_row['country'])
            target_list = index.get(key, [])

            if not target_list:
                candidates[s1_id] = set()
                matches[s1_id] = set()
                continue

            cands = set()
            matched = set()

            # Limit candidates for speed
            for target in target_list[:100]:  # Max 100 candidates per entity
                target_id = target['id']
                cands.add(target_id)

                # Quick filters first
                # 1. Address number must match if both exist
                if s1_row['addr_num'] and target['addr_num']:
                    if s1_row['addr_num'] != target['addr_num']:
                        continue

                # 2. Compute similarities
                name_sim = fuzz.ratio(s1_row['name_norm'], target['name'])

                # Only compute address similarity if name is promising
                if name_sim >= 80:
                    addr_sim = fuzz.ratio(s1_row['addr_norm'], target['addr'])

                    # Match if both are high
                    if addr_sim >= 70:
                        matched.add(target_id)

            candidates[s1_id] = cands
            matches[s1_id] = matched

    # Write outputs
    print("\n" + "="*80)
    print("WRITING OUTPUT FILES")
    print("="*80)

    os.makedirs('../../../output', exist_ok=True)

    print("Writing candidate_pairs.tsv...")
    with open('../../../output/candidate_pairs.tsv', 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1['entity_id']:
            cands = candidates.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")

    print("Writing matching_results.tsv...")
    with open('../../../output/matching_results.tsv', 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1['entity_id']:
            match = matches.get(s1_id, set())
            f.write(f"{s1_id}\t{','.join(sorted(match))}\n")

    # Stats
    total_cands = sum(len(v) for v in candidates.values())
    total_match = sum(len(v) for v in matches.values())
    with_match = sum(1 for v in matches.values() if v)

    print(f"\n=== SUMMARY ===")
    print(f"S1 entities: {len(s1):,}")
    print(f"With matches: {with_match:,}")
    print(f"Singletons: {len(s1) - with_match:,}")
    print(f"Total candidates: {total_cands:,}")
    print(f"Total matches: {total_match:,}")
    print(f"Avg candidates/entity: {total_cands/len(s1):.1f}")
    print(f"Avg matches/entity: {total_match/len(s1):.2f}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
