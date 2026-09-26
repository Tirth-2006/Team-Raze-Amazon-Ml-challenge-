"""
Ultra memory-efficient submission script
Chunks BOTH S1 and S2/S3 to stay under 6GB RAM limit

CHANGES FROM v2:
- S2/S3 also loaded in chunks (not all at once)
- Even smaller S1 chunks (1000 instead of 2000)
- Simplified blocking (2-pass instead of 4-pass to reduce memory)
- Progress tracking to monitor memory usage
"""
import pandas as pd
from rapidfuzz import fuzz
import sys
import os
from collections import defaultdict
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def lightweight_blocking(s1_row, s2_s3_chunk):
    """
    Lightweight 2-pass blocking to reduce memory
    Pass 1: Name prefix (4 chars) + country
    Pass 2: Name token overlap + country
    """
    candidates = set()

    s1_name_norm = normalize_business_name(s1_row['business_name'])
    s1_country = s1_row['country']

    # Pass 1: Name prefix
    name_prefix = s1_name_norm[:4] if len(s1_name_norm) >= 4 else s1_name_norm

    mask = (s2_s3_chunk['country'] == s1_country) & \
           (s2_s3_chunk['name_norm'].str.startswith(name_prefix))

    for idx in s2_s3_chunk[mask].index:
        candidates.add(s2_s3_chunk.loc[idx, 'entity_id'])

    # Pass 2: Name token overlap (only check if few candidates so far)
    if len(candidates) < 20:
        s1_tokens = set(t for t in s1_name_norm.split() if len(t) >= 3)
        if s1_tokens:
            for idx, row in s2_s3_chunk[s2_s3_chunk['country'] == s1_country].iterrows():
                t_tokens = set(t for t in row['name_norm'].split() if len(t) >= 3)
                if s1_tokens & t_tokens:
                    candidates.add(row['entity_id'])

    return candidates


def rank_and_match(s1_row, candidates_dict, s2_s3_chunk, name_thresh=85, addr_thresh=70, max_cand=50):
    """
    Rank candidates and find matches
    Returns: (candidate_set, match_list_with_confidence)
    """
    s1_name_norm = normalize_business_name(s1_row['business_name'])
    s1_addr_norm = normalize_address(s1_row['business_address'])

    # Score candidates
    scored = []
    for cand_id in candidates_dict:
        if cand_id not in s2_s3_chunk.index:
            continue

        t_row = s2_s3_chunk.loc[cand_id]

        # Quick score
        name_sim = fuzz.ratio(s1_name_norm, t_row['name_norm']) / 100.0
        addr_sim = fuzz.ratio(s1_addr_norm, t_row['addr_norm']) / 100.0
        quick_score = 0.6 * name_sim + 0.4 * addr_sim

        scored.append((cand_id, quick_score, name_sim * 100, addr_sim * 100))

    # Sort and take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored[:max_cand]

    # Find matches
    matches = []
    candidate_ids = set()

    for cand_id, quick_score, name_sim, addr_sim in top_candidates:
        candidate_ids.add(cand_id)

        if name_sim >= name_thresh and addr_sim >= addr_thresh:
            confidence = name_sim + addr_sim
            matches.append((cand_id, confidence))

    return candidate_ids, matches


def process_s1_chunk_memory_safe(s1_chunk, s2_path, s3_path, chunk_size_s2s3=100000):
    """
    Process S1 chunk against S2/S3 loaded in smaller chunks
    """
    print(f"  Processing {len(s1_chunk)} S1 entities...")

    # Normalize S1 once
    s1_chunk = s1_chunk.copy()
    s1_chunk['name_norm'] = s1_chunk['business_name'].apply(normalize_business_name)
    s1_chunk['addr_norm'] = s1_chunk['business_address'].apply(normalize_address)

    # Accumulate candidates across S2/S3 chunks
    all_candidates = defaultdict(set)  # s1_id -> set of candidate ids
    all_candidate_data = {}  # candidate_id -> row data

    print("  Loading and blocking S2 in chunks...")
    chunk_num = 0
    for s2_chunk in pd.read_csv(s2_path, sep='\t', chunksize=chunk_size_s2s3):
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"    S2 chunk {chunk_num}...")

        s2_chunk['name_norm'] = s2_chunk['business_name'].apply(normalize_business_name)
        s2_chunk['addr_norm'] = s2_chunk['business_address'].apply(normalize_address)
        s2_chunk = s2_chunk.set_index('entity_id')

        for _, s1_row in s1_chunk.iterrows():
            cands = lightweight_blocking(s1_row, s2_chunk)
            all_candidates[s1_row['entity_id']].update(cands)

            # Store candidate data
            for cid in cands:
                if cid not in all_candidate_data:
                    all_candidate_data[cid] = s2_chunk.loc[cid].to_dict()

        del s2_chunk
        gc.collect()

    print("  Loading and blocking S3 in chunks...")
    chunk_num = 0
    for s3_chunk in pd.read_csv(s3_path, sep='\t', chunksize=chunk_size_s2s3):
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"    S3 chunk {chunk_num}...")

        s3_chunk['name_norm'] = s3_chunk['business_name'].apply(normalize_business_name)
        s3_chunk['addr_norm'] = s3_chunk['business_address'].apply(normalize_address)
        s3_chunk = s3_chunk.set_index('entity_id')

        for _, s1_row in s1_chunk.iterrows():
            cands = lightweight_blocking(s1_row, s3_chunk)
            all_candidates[s1_row['entity_id']].update(cands)

            for cid in cands:
                if cid not in all_candidate_data:
                    all_candidate_data[cid] = s3_chunk.loc[cid].to_dict()

        del s3_chunk
        gc.collect()

    # Now rank and match
    print("  Ranking candidates and finding matches...")

    # Convert candidate data to DataFrame for easier lookup
    cand_df = pd.DataFrame.from_dict(all_candidate_data, orient='index')

    results = []
    for _, s1_row in s1_chunk.iterrows():
        s1_id = s1_row['entity_id']

        if s1_id not in all_candidates or not all_candidates[s1_id]:
            results.append({
                's1_id': s1_id,
                'candidates': set(),
                'matches': []
            })
            continue

        cand_ids, matches = rank_and_match(
            s1_row, all_candidates[s1_id], cand_df
        )

        results.append({
            's1_id': s1_id,
            'candidates': cand_ids,
            'matches': matches
        })

    del cand_df
    gc.collect()

    return results


def resolve_conflicts(all_results):
    """Conflict resolution"""
    print("\nResolving conflicts...")

    target_claims = defaultdict(list)

    for result in all_results:
        s1_id = result['s1_id']
        for target_id, confidence in result['matches']:
            target_claims[target_id].append((s1_id, confidence))

    conflicts = {tid: claims for tid, claims in target_claims.items() if len(claims) > 1}
    print(f"  Found {len(conflicts):,} contested targets")

    if not conflicts:
        for result in all_results:
            result['matches'] = set(tid for tid, _ in result['matches'])
        return all_results

    winners = {}
    for target_id, claims in conflicts.items():
        claims.sort(key=lambda x: x[1], reverse=True)
        winners[target_id] = claims[0][0]

    removed = 0
    for result in all_results:
        s1_id = result['s1_id']
        resolved_matches = []

        for target_id, confidence in result['matches']:
            if target_id in winners:
                if winners[target_id] == s1_id:
                    resolved_matches.append(target_id)
                else:
                    removed += 1
            else:
                resolved_matches.append(target_id)

        result['matches'] = set(resolved_matches)

    print(f"  Removed {removed:,} matches due to conflicts")
    return all_results


def main():
    print("="*80)
    print("ULTRA MEMORY-EFFICIENT SUBMISSION (v3)")
    print("="*80)
    print("\nOptimizations:")
    print("  - S2/S3 loaded in 100K chunks (not all at once)")
    print("  - S1 chunks of 1000 entities")
    print("  - 2-pass blocking (reduced from 4-pass)")
    print("  - Aggressive garbage collection")
    print("="*80)

    base = '../../../student_resource/dataset/test'
    s2_path = f'{base}/test_source2.tsv'
    s3_path = f'{base}/test_source3.tsv'

    # Count S1
    print("\nCounting S1 entities...")
    s1_count = sum(1 for _ in open(f'{base}/test_source1.tsv', encoding='utf-8')) - 1
    print(f"Total S1: {s1_count:,}")

    s1_chunk_size = 1000
    all_results = []
    processed = 0

    print(f"\nProcessing S1 in chunks of {s1_chunk_size:,}...")

    for s1_chunk in pd.read_csv(f'{base}/test_source1.tsv', sep='\t', chunksize=s1_chunk_size):
        chunk_size = len(s1_chunk)
        processed += chunk_size

        print(f"\n[{processed:,} / {s1_count:,}] Processing entities {processed-chunk_size+1:,} to {processed:,}")

        chunk_results = process_s1_chunk_memory_safe(s1_chunk, s2_path, s3_path)
        all_results.extend(chunk_results)

        gc.collect()

    print(f"\nCompleted all {processed:,} entities")

    # Conflict resolution
    all_results = resolve_conflicts(all_results)

    # Write output
    os.makedirs('../../../output', exist_ok=True)

    print("\nWriting output...")
    with open('../../../output/candidate_pairs.tsv', 'w', encoding='utf-8') as f_cand, \
         open('../../../output/matching_results.tsv', 'w', encoding='utf-8') as f_match:

        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for result in all_results:
            s1_id = result['s1_id']
            cands = ','.join(sorted(result['candidates']))
            matches = ','.join(sorted(result['matches']))

            f_cand.write(f"{s1_id}\t{cands}\n")
            f_match.write(f"{s1_id}\t{matches}\n")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - output/candidate_pairs.tsv")
    print("  - output/matching_results.tsv")


if __name__ == '__main__':
    main()
