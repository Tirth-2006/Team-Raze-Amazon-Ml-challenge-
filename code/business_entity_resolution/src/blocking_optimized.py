"""
Optimized blocking strategy - focus on precision to reduce candidate set size
"""
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, Set, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address


def generate_blocking_key_name_first_token(name: str) -> str:
    """First significant token (length >= 3) from normalized name"""
    normalized = normalize_business_name(name)
    tokens = normalized.split()
    for token in tokens:
        if len(token) >= 3:
            return token
    return normalized.split()[0] if normalized.split() else ""


def generate_blocking_key_name_sorted_tokens(name: str, top_n: int = 3) -> str:
    """Sorted concatenation of top N tokens (by length)"""
    normalized = normalize_business_name(name)
    tokens = [t for t in normalized.split() if len(t) >= 3]
    tokens_sorted = sorted(tokens, key=len, reverse=True)[:top_n]
    return '|'.join(sorted(tokens_sorted))


def generate_blocking_key_address_number_prefix(address: str) -> str:
    """First number + first significant word"""
    normalized = normalize_address(address)
    numbers = re.findall(r'\d+', normalized)
    tokens = [t for t in normalized.split() if len(t) >= 3 and not t.isdigit()]

    if numbers and tokens:
        return f"{numbers[0]}_{tokens[0]}"
    elif numbers:
        return numbers[0]
    elif tokens:
        return tokens[0]
    return ""


def generate_candidates_optimized(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Set[str]]:
    """
    Optimized multi-pass blocking with tighter keys

    Passes (all with country filter):
    1. First significant name token + country
    2. Sorted top-3 name tokens + country
    3. Address number + first address token + country
    """
    candidates = defaultdict(set)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    if verbose:
        print(f"Blocking: {len(s1_df):,} S1 entities vs {len(target_df):,} S2+S3 targets")

    # PASS 1: First significant name token + country
    if verbose:
        print("\nPass 1: First significant name token + country")

    s1_index = defaultdict(list)
    for _, row in s1_df.iterrows():
        key = (generate_blocking_key_name_first_token(row['business_name']), row['country'])
        if key[0]:  # Skip empty keys
            s1_index[key].append(row['entity_id'])

    target_index = defaultdict(list)
    for _, row in target_df.iterrows():
        key = (generate_blocking_key_name_first_token(row['business_name']), row['country'])
        if key[0]:
            target_index[key].append(row['entity_id'])

    pass1_pairs = 0
    for key in s1_index:
        if key in target_index:
            for s1_id in s1_index[key]:
                candidates[s1_id].update(target_index[key])
                pass1_pairs += len(target_index[key])

    if verbose:
        print(f"  Generated {pass1_pairs:,} candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates")

    # PASS 2: Sorted top-3 name tokens + country
    if verbose:
        print("\nPass 2: Sorted top-3 name tokens + country")

    s1_sorted_tokens = defaultdict(list)
    for _, row in s1_df.iterrows():
        key = (generate_blocking_key_name_sorted_tokens(row['business_name'], 3), row['country'])
        if key[0]:
            s1_sorted_tokens[key].append(row['entity_id'])

    target_sorted_tokens = defaultdict(list)
    for _, row in target_df.iterrows():
        key = (generate_blocking_key_name_sorted_tokens(row['business_name'], 3), row['country'])
        if key[0]:
            target_sorted_tokens[key].append(row['entity_id'])

    pass2_pairs = 0
    for key in s1_sorted_tokens:
        if key in target_sorted_tokens:
            for s1_id in s1_sorted_tokens[key]:
                new_candidates = set(target_sorted_tokens[key]) - candidates[s1_id]
                candidates[s1_id].update(new_candidates)
                pass2_pairs += len(new_candidates)

    if verbose:
        print(f"  Generated {pass2_pairs:,} NEW candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates")

    # PASS 3: Address number + first token + country
    if verbose:
        print("\nPass 3: Address number + first token + country")

    s1_addr_key = defaultdict(list)
    for _, row in s1_df.iterrows():
        key = (generate_blocking_key_address_number_prefix(row['business_address']), row['country'])
        if key[0]:
            s1_addr_key[key].append(row['entity_id'])

    target_addr_key = defaultdict(list)
    for _, row in target_df.iterrows():
        key = (generate_blocking_key_address_number_prefix(row['business_address']), row['country'])
        if key[0]:
            target_addr_key[key].append(row['entity_id'])

    pass3_pairs = 0
    for key in s1_addr_key:
        if key in target_addr_key:
            for s1_id in s1_addr_key[key]:
                new_candidates = set(target_addr_key[key]) - candidates[s1_id]
                candidates[s1_id].update(new_candidates)
                pass3_pairs += len(new_candidates)

    if verbose:
        print(f"  Generated {pass3_pairs:,} NEW candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates")

    # Ensure every S1 entity has an entry
    for _, row in s1_df.iterrows():
        if row['entity_id'] not in candidates:
            candidates[row['entity_id']] = set()

    # Summary
    if verbose:
        total_pairs = sum(len(v) for v in candidates.values())
        non_empty = sum(1 for v in candidates.values() if len(v) > 0)
        avg_candidates = total_pairs / len(candidates) if candidates else 0

        print(f"\n=== BLOCKING SUMMARY ===")
        print(f"Total S1 entities: {len(candidates):,}")
        print(f"S1 entities with candidates: {non_empty:,}")
        print(f"S1 entities with NO candidates: {len(candidates) - non_empty:,}")
        print(f"Total candidate pairs: {total_pairs:,}")
        print(f"Average candidates per S1 entity: {avg_candidates:.1f}")

        import numpy as np
        candidate_counts = [len(v) for v in candidates.values()]
        print(f"Median candidates per entity: {np.median(candidate_counts):.0f}")
        print(f"95th percentile: {np.percentile(candidate_counts, 95):.0f}")

    return dict(candidates)


if __name__ == '__main__':
    print("Testing optimized blocking on small sample...")

    s1 = pd.read_csv('../../../student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=1000)
    s2 = pd.read_csv('../../../student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=5000)
    s3 = pd.read_csv('../../../student_resource/dataset/train/train_source3.tsv', sep='\t', nrows=5000)

    candidates_opt = generate_candidates_optimized(s1, s2, s3, verbose=True)

    # Load ground truth for this sample
    gt = pd.read_csv('../../../student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
    gt_sample = gt[gt['source1_entity_id'].isin(s1['entity_id'])]

    # Calculate recall
    total_found = 0
    total_true = 0

    for _, row in gt_sample.iterrows():
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == '':
            continue

        true_matches = set(row['matched_entity_ids'].split(','))
        candidate_set = candidates_opt.get(s1_id, set())

        found = len(true_matches & candidate_set)
        total_found += found
        total_true += len(true_matches)

    recall = total_found / total_true if total_true > 0 else 0
    print(f"\n=== RECALL ON SAMPLE ===")
    print(f"Found {total_found:,} / {total_true:,} true matches")
    print(f"Recall: {recall:.4f}")
