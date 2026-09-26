"""
Multi-pass blocking strategy to generate candidate pairs

Goal: High recall (catch true matches) with manageable candidate set size
Each pass generates candidates using different keys, then union them
"""
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, Set, Tuple
from normalization import normalize_business_name, normalize_address, extract_tokens, extract_numeric_tokens


def generate_blocking_key_name_prefix(name: str, n: int = 4) -> str:
    """First n characters of normalized name"""
    normalized = normalize_business_name(name)
    if len(normalized) < n:
        return normalized
    return normalized[:n]


def generate_blocking_key_name_tokens(name: str) -> Set[str]:
    """Set of significant tokens from name (length >= 3)"""
    normalized = normalize_business_name(name)
    tokens = normalized.split()
    # Filter out very short tokens (likely articles/prepositions)
    return {t for t in tokens if len(t) >= 3}


def generate_blocking_key_address_numeric(address: str) -> str:
    """Extract first significant number from address (street number)"""
    normalized = normalize_address(address)
    numbers = re.findall(r'\d+', normalized)
    if numbers:
        return numbers[0]
    return ""


def generate_blocking_key_address_tokens(address: str) -> Set[str]:
    """Significant tokens from address"""
    normalized = normalize_address(address)
    tokens = normalized.split()
    # Keep tokens length >= 3
    return {t for t in tokens if len(t) >= 3 and not t.isdigit()}


def build_inverted_index(df: pd.DataFrame, key_func, multi_value: bool = False) -> Dict:
    """
    Build inverted index: blocking_key -> list of entity_ids

    Args:
        df: DataFrame with entity_id column
        key_func: Function to generate blocking key from row
        multi_value: If True, key_func returns a set of keys per row
    """
    index = defaultdict(list)

    for _, row in df.iterrows():
        entity_id = row['entity_id']

        if multi_value:
            keys = key_func(row)
            for key in keys:
                if key:  # Skip empty keys
                    index[key].append(entity_id)
        else:
            key = key_func(row)
            if key:  # Skip empty keys
                index[key].append(entity_id)

    return dict(index)


def generate_candidates_multi_pass(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Set[str]]:
    """
    Multi-pass blocking to generate candidates

    Passes:
    1. Name prefix (first 4 chars) + country
    2. Name token overlap (any shared significant token) + country
    3. Address numeric (first number) + country
    4. Address token overlap + country

    Returns:
        Dict mapping S1 entity_id -> set of candidate S2/S3 entity_ids
    """
    candidates = defaultdict(set)

    # Combine S2 and S3 into single target pool
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    if verbose:
        print(f"Blocking: {len(s1_df):,} S1 entities vs {len(target_df):,} S2+S3 targets")

    # PASS 1: Name prefix + country
    if verbose:
        print("\nPass 1: Name prefix (4 chars) + country")

    s1_index = {}
    for _, row in s1_df.iterrows():
        key = (generate_blocking_key_name_prefix(row['business_name'], 4), row['country'])
        if key not in s1_index:
            s1_index[key] = []
        s1_index[key].append(row['entity_id'])

    target_index = {}
    for _, row in target_df.iterrows():
        key = (generate_blocking_key_name_prefix(row['business_name'], 4), row['country'])
        if key not in target_index:
            target_index[key] = []
        target_index[key].append(row['entity_id'])

    pass1_pairs = 0
    for key in s1_index:
        if key in target_index:
            for s1_id in s1_index[key]:
                candidates[s1_id].update(target_index[key])
                pass1_pairs += len(target_index[key])

    if verbose:
        print(f"  Generated {pass1_pairs:,} candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates so far")

    # PASS 2: Name token overlap + country
    if verbose:
        print("\nPass 2: Name token overlap + country")

    s1_tokens = {}
    for _, row in s1_df.iterrows():
        tokens = generate_blocking_key_name_tokens(row['business_name'])
        country = row['country']
        for token in tokens:
            key = (token, country)
            if key not in s1_tokens:
                s1_tokens[key] = []
            s1_tokens[key].append(row['entity_id'])

    target_tokens = {}
    for _, row in target_df.iterrows():
        tokens = generate_blocking_key_name_tokens(row['business_name'])
        country = row['country']
        for token in tokens:
            key = (token, country)
            if key not in target_tokens:
                target_tokens[key] = []
            target_tokens[key].append(row['entity_id'])

    pass2_pairs = 0
    for key in s1_tokens:
        if key in target_tokens:
            for s1_id in s1_tokens[key]:
                new_candidates = set(target_tokens[key]) - candidates[s1_id]
                candidates[s1_id].update(new_candidates)
                pass2_pairs += len(new_candidates)

    if verbose:
        print(f"  Generated {pass2_pairs:,} NEW candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates so far")

    # PASS 3: Address numeric + country
    if verbose:
        print("\nPass 3: Address numeric (first number) + country")

    s1_addr_num = {}
    for _, row in s1_df.iterrows():
        num = generate_blocking_key_address_numeric(row['business_address'])
        if num:
            key = (num, row['country'])
            if key not in s1_addr_num:
                s1_addr_num[key] = []
            s1_addr_num[key].append(row['entity_id'])

    target_addr_num = {}
    for _, row in target_df.iterrows():
        num = generate_blocking_key_address_numeric(row['business_address'])
        if num:
            key = (num, row['country'])
            if key not in target_addr_num:
                target_addr_num[key] = []
            target_addr_num[key].append(row['entity_id'])

    pass3_pairs = 0
    for key in s1_addr_num:
        if key in target_addr_num:
            for s1_id in s1_addr_num[key]:
                new_candidates = set(target_addr_num[key]) - candidates[s1_id]
                candidates[s1_id].update(new_candidates)
                pass3_pairs += len(new_candidates)

    if verbose:
        print(f"  Generated {pass3_pairs:,} NEW candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates so far")

    # PASS 4: Address token overlap + country
    if verbose:
        print("\nPass 4: Address token overlap + country")

    s1_addr_tokens = {}
    for _, row in s1_df.iterrows():
        tokens = generate_blocking_key_address_tokens(row['business_address'])
        country = row['country']
        for token in tokens:
            key = (token, country)
            if key not in s1_addr_tokens:
                s1_addr_tokens[key] = []
            s1_addr_tokens[key].append(row['entity_id'])

    target_addr_tokens = {}
    for _, row in target_df.iterrows():
        tokens = generate_blocking_key_address_tokens(row['business_address'])
        country = row['country']
        for token in tokens:
            key = (token, country)
            if key not in target_addr_tokens:
                target_addr_tokens[key] = []
            target_addr_tokens[key].append(row['entity_id'])

    pass4_pairs = 0
    for key in s1_addr_tokens:
        if key in target_addr_tokens:
            for s1_id in s1_addr_tokens[key]:
                new_candidates = set(target_addr_tokens[key]) - candidates[s1_id]
                candidates[s1_id].update(new_candidates)
                pass4_pairs += len(target_addr_tokens[key])

    if verbose:
        print(f"  Generated {pass4_pairs:,} NEW candidate pairs")
        print(f"  {len(candidates):,} S1 entities have candidates so far")

    # Ensure every S1 entity has an entry (even if empty)
    for _, row in s1_df.iterrows():
        if row['entity_id'] not in candidates:
            candidates[row['entity_id']] = set()

    # Summary statistics
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

        # Distribution
        candidate_counts = [len(v) for v in candidates.values()]
        import numpy as np
        print(f"Median candidates per entity: {np.median(candidate_counts):.0f}")
        print(f"95th percentile: {np.percentile(candidate_counts, 95):.0f}")
        print(f"Max candidates for any entity: {max(candidate_counts):,}")

    return dict(candidates)
