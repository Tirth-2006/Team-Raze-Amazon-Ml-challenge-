"""
Memory-efficient full test set processing
Processes in chunks to avoid memory issues
"""
import pandas as pd
from rapidfuzz import fuzz
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name


def process_s1_chunk(s1_chunk, s2_chunk, s3_chunk, name_thresh=85, addr_thresh=70):
    """Process one chunk of S1 entities"""
    from normalization import normalize_address

    # Normalize chunk
    s1_chunk = s1_chunk.copy()
    s1_chunk['name_norm'] = s1_chunk['business_name'].apply(normalize_business_name)
    s1_chunk['addr_norm'] = s1_chunk['business_address'].apply(normalize_address)

    s2_chunk = s2_chunk.copy()
    s2_chunk['name_norm'] = s2_chunk['business_name'].apply(normalize_business_name)
    s2_chunk['addr_norm'] = s2_chunk['business_address'].apply(normalize_address)

    s3_chunk = s3_chunk.copy()
    s3_chunk['name_norm'] = s3_chunk['business_name'].apply(normalize_business_name)
    s3_chunk['addr_norm'] = s3_chunk['business_address'].apply(normalize_address)

    target = pd.concat([s2_chunk, s3_chunk], ignore_index=True)

    results = []

    for _, s1_row in s1_chunk.iterrows():
        s1_id = s1_row['entity_id']

        # Filter by country and name prefix
        name_prefix = s1_row['name_norm'][:3] if len(s1_row['name_norm']) >= 3 else s1_row['name_norm']

        filtered = target[
            (target['country'] == s1_row['country']) &
            (target['name_norm'].str.startswith(name_prefix))
        ].head(50)  # Limit to 50 candidates

        candidates = set()
        matches = set()

        for _, t_row in filtered.iterrows():
            tid = t_row['entity_id']
            candidates.add(tid)

            # Compute similarities
            name_sim = fuzz.ratio(s1_row['name_norm'], t_row['name_norm'])

            if name_sim >= name_thresh:
                addr_sim = fuzz.ratio(s1_row['addr_norm'], t_row['addr_norm'])
                if addr_sim >= addr_thresh:
                    matches.add(tid)

        results.append({
            's1_id': s1_id,
            'candidates': candidates,
            'matches': matches
        })

    return results


def main():
    print("="*80)
    print("MEMORY-EFFICIENT FULL TEST PROCESSING")
    print("="*80)

    base = '../../../student_resource/dataset/test'

    # Get S1 size
    print("\nCounting S1 entities...")
    s1_count = sum(1 for _ in open(f'{base}/test_source1.tsv', encoding='utf-8')) - 1
    print(f"Total S1 entities: {s1_count:,}")

    # Process in chunks
    s1_chunk_size = 5000  # Process 5K S1 entities at a time

    # Load S2 and S3 once (they're the lookup set)
    print("\nLoading S2 and S3...")
    s2 = pd.read_csv(f'{base}/test_source2.tsv', sep='\t')
    s3 = pd.read_csv(f'{base}/test_source3.tsv', sep='\t')
    print(f"S2: {len(s2):,}, S3: {len(s3):,}")

    # Open output files
    os.makedirs('../../../output', exist_ok=True)

    f_cand = open('../../../output/candidate_pairs.tsv', 'w', encoding='utf-8')
    f_match = open('../../../output/matching_results.tsv', 'w', encoding='utf-8')

    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # Process S1 in chunks
    print(f"\nProcessing S1 in chunks of {s1_chunk_size:,}...")

    processed = 0

    for chunk in pd.read_csv(f'{base}/test_source1.tsv', sep='\t', chunksize=s1_chunk_size):
        chunk_size = len(chunk)
        processed += chunk_size

        print(f"  Processing entities {processed-chunk_size+1:,} to {processed:,}...")

        # Process this chunk
        results = process_s1_chunk(chunk, s2, s3)

        # Write results immediately
        for result in results:
            s1_id = result['s1_id']
            cands = ','.join(sorted(result['candidates']))
            matches = ','.join(sorted(result['matches']))

            f_cand.write(f"{s1_id}\t{cands}\n")
            f_match.write(f"{s1_id}\t{matches}\n")

        f_cand.flush()
        f_match.flush()

    f_cand.close()
    f_match.close()

    print(f"\n  Completed all {processed:,} entities")
    print("\n" + "="*80)
    print("FILES WRITTEN SUCCESSFULLY")
    print("="*80)
    print("\nOutput files:")
    print("  - output/candidate_pairs.tsv")
    print("  - output/matching_results.tsv")
    print("\nNext: Run validation")


if __name__ == '__main__':
    main()
