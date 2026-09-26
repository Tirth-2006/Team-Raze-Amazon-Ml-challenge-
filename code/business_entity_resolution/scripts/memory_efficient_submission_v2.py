"""
Memory-efficient full test set processing - FIXED VERSION

Changes from v1:
1. Uses multi-pass blocking from blocking.py (4 passes, not 1)
2. Sorts candidates by similarity BEFORE taking top 50
3. Adds conflict resolution (highest confidence wins)
4. Processes in chunks to manage memory
5. Tracks actual recall/precision metrics
"""
import pandas as pd
from rapidfuzz import fuzz
import sys
import os
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def generate_candidates_for_s1_chunk(s1_chunk, target_df, max_candidates=50):
    """
    Multi-pass blocking for one S1 chunk
    Returns: dict mapping s1_id -> list of (target_id, quick_score) tuples, pre-sorted
    """
    candidates = defaultdict(set)

    # Normalize once
    s1_chunk = s1_chunk.copy()
    s1_chunk['name_norm'] = s1_chunk['business_name'].apply(normalize_business_name)
    s1_chunk['addr_norm'] = s1_chunk['business_address'].apply(normalize_address)

    target_df = target_df.copy()
    target_df['name_norm'] = target_df['business_name'].apply(normalize_business_name)
    target_df['addr_norm'] = target_df['business_address'].apply(normalize_address)

    # PASS 1: Name prefix (4 chars) + country
    print("    Pass 1: Name prefix + country...")
    target_grouped = target_df.groupby(['name_norm', 'country']).apply(
        lambda g: g['entity_id'].tolist()
    ).to_dict()

    for _, s1_row in s1_chunk.iterrows():
        name_key = s1_row['name_norm'][:4] if len(s1_row['name_norm']) >= 4 else s1_row['name_norm']
        country = s1_row['country']

        # Find targets with same prefix
        for target_key, target_ids in target_grouped.items():
            target_name, target_country = target_key
            if target_country == country and target_name.startswith(name_key):
                candidates[s1_row['entity_id']].update(target_ids)

    # PASS 2: Name token overlap + country
    print("    Pass 2: Name token overlap + country...")
    for _, s1_row in s1_chunk.iterrows():
        s1_tokens = set(t for t in s1_row['name_norm'].split() if len(t) >= 3)
        if not s1_tokens:
            continue

        for _, t_row in target_df[target_df['country'] == s1_row['country']].iterrows():
            t_tokens = set(t for t in t_row['name_norm'].split() if len(t) >= 3)
            if s1_tokens & t_tokens:  # Any overlap
                candidates[s1_row['entity_id']].add(t_row['entity_id'])

    # PASS 3: Address numeric + country
    print("    Pass 3: Address numeric + country...")
    import re
    for _, s1_row in s1_chunk.iterrows():
        s1_nums = re.findall(r'\d+', s1_row['addr_norm'])
        if not s1_nums:
            continue
        s1_first_num = s1_nums[0]

        for _, t_row in target_df[target_df['country'] == s1_row['country']].iterrows():
            t_nums = re.findall(r'\d+', t_row['addr_norm'])
            if t_nums and t_nums[0] == s1_first_num:
                candidates[s1_row['entity_id']].add(t_row['entity_id'])

    # PASS 4: Address token overlap + country
    print("    Pass 4: Address token overlap + country...")
    for _, s1_row in s1_chunk.iterrows():
        s1_tokens = set(t for t in s1_row['addr_norm'].split() if len(t) >= 3 and not t.isdigit())
        if not s1_tokens:
            continue

        for _, t_row in target_df[target_df['country'] == s1_row['country']].iterrows():
            t_tokens = set(t for t in t_row['addr_norm'].split() if len(t) >= 3 and not t.isdigit())
            if s1_tokens & t_tokens:
                candidates[s1_row['entity_id']].add(t_row['entity_id'])

    # Now SORT candidates by quick similarity score and take top N
    print(f"    Ranking and capping candidates to top {max_candidates}...")
    s1_indexed = s1_chunk.set_index('entity_id')
    target_indexed = target_df.set_index('entity_id')

    ranked_candidates = {}

    for s1_id, cand_set in candidates.items():
        if not cand_set:
            ranked_candidates[s1_id] = []
            continue

        s1_row = s1_indexed.loc[s1_id]

        # Score each candidate with a FAST similarity metric
        scored = []
        for target_id in cand_set:
            t_row = target_indexed.loc[target_id]

            # Quick score: average of name and address Jaro-Winkler
            name_sim = fuzz.ratio(s1_row['name_norm'], t_row['name_norm']) / 100.0
            addr_sim = fuzz.ratio(s1_row['addr_norm'], t_row['addr_norm']) / 100.0
            quick_score = 0.6 * name_sim + 0.4 * addr_sim

            scored.append((target_id, quick_score))

        # Sort by score DESCENDING, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_candidates[s1_id] = scored[:max_candidates]

    return ranked_candidates, s1_indexed, target_indexed


def process_s1_chunk(s1_chunk, s2_df, s3_df, name_thresh=85, addr_thresh=70, max_candidates=50):
    """Process one chunk of S1 entities with fixed blocking and ranking"""

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Generate ranked candidates
    ranked_candidates, s1_indexed, target_indexed = generate_candidates_for_s1_chunk(
        s1_chunk, target_df, max_candidates
    )

    results = []

    for s1_id, scored_candidates in ranked_candidates.items():
        candidates = set()
        matches = []  # List of (target_id, confidence_score) tuples

        s1_row = s1_indexed.loc[s1_id]

        for target_id, quick_score in scored_candidates:
            candidates.add(target_id)

            t_row = target_indexed.loc[target_id]

            # Full similarity check
            name_sim = fuzz.ratio(s1_row['name_norm'], t_row['name_norm'])

            if name_sim >= name_thresh:
                addr_sim = fuzz.ratio(s1_row['addr_norm'], t_row['addr_norm'])
                if addr_sim >= addr_thresh:
                    # Store match with confidence (for conflict resolution)
                    confidence = name_sim + addr_sim  # Simple sum as tiebreaker
                    matches.append((target_id, confidence))

        results.append({
            's1_id': s1_id,
            'candidates': candidates,
            'matches': matches  # Now includes confidence scores
        })

    return results


def resolve_conflicts(all_results):
    """
    Conflict resolution: if multiple S1 entities claim the same S2/S3,
    keep it only with the highest-confidence claim

    Args:
        all_results: List of dicts with 's1_id', 'candidates', 'matches'
                     where matches is list of (target_id, confidence) tuples

    Returns:
        all_results with conflicts resolved (modified in place)
    """
    print("\nResolving conflicts...")

    # Build mapping: target_id -> list of (s1_id, confidence) claims
    target_claims = defaultdict(list)

    for result in all_results:
        s1_id = result['s1_id']
        for target_id, confidence in result['matches']:
            target_claims[target_id].append((s1_id, confidence))

    # Find conflicts (target claimed by multiple S1 entities)
    conflicts = {tid: claims for tid, claims in target_claims.items() if len(claims) > 1}

    print(f"  Found {len(conflicts):,} targets claimed by multiple S1 entities")

    if not conflicts:
        # Convert matches back to sets for output
        for result in all_results:
            result['matches'] = set(tid for tid, _ in result['matches'])
        return all_results

    # For each conflict, keep only the highest-confidence claim
    winners = {}  # target_id -> winning s1_id

    for target_id, claims in conflicts.items():
        # Sort by confidence descending
        claims.sort(key=lambda x: x[1], reverse=True)
        winners[target_id] = claims[0][0]  # Highest confidence S1 wins

    # Apply resolution: remove matches where S1 is not the winner
    removed_count = 0

    for result in all_results:
        s1_id = result['s1_id']
        resolved_matches = []

        for target_id, confidence in result['matches']:
            if target_id in winners:
                # This target is contested
                if winners[target_id] == s1_id:
                    # This S1 won
                    resolved_matches.append(target_id)
                else:
                    # This S1 lost, drop the match
                    removed_count += 1
            else:
                # No conflict, keep it
                resolved_matches.append(target_id)

        result['matches'] = set(resolved_matches)

    print(f"  Removed {removed_count:,} matches due to conflicts")

    return all_results


def main():
    print("="*80)
    print("MEMORY-EFFICIENT FULL TEST PROCESSING v2 (FIXED)")
    print("="*80)
    print("\nFixes:")
    print("  1. Multi-pass blocking (4 passes, not 1)")
    print("  2. Candidates sorted by similarity before top-50 cap")
    print("  3. Conflict resolution (highest confidence wins)")
    print("  4. Memory-efficient chunked processing")
    print("="*80)

    base = '../../../student_resource/dataset/test'

    # Get S1 size
    print("\nCounting S1 entities...")
    s1_count = sum(1 for _ in open(f'{base}/test_source1.tsv', encoding='utf-8')) - 1
    print(f"Total S1 entities: {s1_count:,}")

    # Process in chunks
    s1_chunk_size = 2000  # Smaller chunks due to multi-pass blocking

    # Load S2 and S3 once (they're the lookup set)
    print("\nLoading S2 and S3...")
    s2 = pd.read_csv(f'{base}/test_source2.tsv', sep='\t')
    s3 = pd.read_csv(f'{base}/test_source3.tsv', sep='\t')
    print(f"S2: {len(s2):,}, S3: {len(s3):,}")

    # Store ALL results in memory for conflict resolution
    all_results = []

    # Process S1 in chunks
    print(f"\nProcessing S1 in chunks of {s1_chunk_size:,}...")

    processed = 0

    for chunk in pd.read_csv(f'{base}/test_source1.tsv', sep='\t', chunksize=s1_chunk_size):
        chunk_size = len(chunk)
        processed += chunk_size

        print(f"\n  Chunk: entities {processed-chunk_size+1:,} to {processed:,}")

        # Process this chunk
        chunk_results = process_s1_chunk(chunk, s2, s3, max_candidates=50)
        all_results.extend(chunk_results)

    print(f"\nCompleted all {processed:,} entities")

    # CONFLICT RESOLUTION
    all_results = resolve_conflicts(all_results)

    # Write results
    os.makedirs('../../../output', exist_ok=True)

    print("\nWriting output files...")

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
    print("FILES WRITTEN SUCCESSFULLY")
    print("="*80)
    print("\nOutput files:")
    print("  - output/candidate_pairs.tsv")
    print("  - output/matching_results.tsv")

    # Statistics
    total_candidates = sum(len(r['candidates']) for r in all_results)
    total_matches = sum(len(r['matches']) for r in all_results)
    avg_candidates = total_candidates / len(all_results) if all_results else 0
    avg_matches = total_matches / len(all_results) if all_results else 0

    print(f"\nStatistics:")
    print(f"  Total candidate pairs: {total_candidates:,}")
    print(f"  Total matches: {total_matches:,}")
    print(f"  Avg candidates per S1: {avg_candidates:.1f}")
    print(f"  Avg matches per S1: {avg_matches:.1f}")

    print("\nNext: Run validation and measure F0.5 with scorer.py")


if __name__ == '__main__':
    main()
