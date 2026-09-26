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
    verbose: bool = True,
    target_chunk_size: int = 100_000,
    max_candidates_per_entity: int = 5_000,
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

    ``target_chunk_size`` bounds the target-side scan. ``max_candidates_per_entity``
    is a deterministic safety cap that prevents a common blocking key from
    creating an unbounded Python set.
    """
    candidates = defaultdict(set)
    # Do not concatenate or index the target pool.  On the full data this can
    # be tens of millions of rows and the old target-side dictionaries alone
    # could exhaust the process before matching started.
    target_frames = (s2_df, s3_df)

    if verbose:
        print(
            f"Blocking: {len(s1_df):,} S1 entities vs "
            f"{sum(len(frame) for frame in target_frames):,} S2+S3 targets "
            f"(target chunks of {target_chunk_size:,})"
        )

    passes = (
        ("Name prefix (4 chars)", lambda row: generate_blocking_key_name_prefix(row.business_name, 4)),
        ("Name token overlap", lambda row: generate_blocking_key_name_tokens(row.business_name)),
        ("Address numeric", lambda row: generate_blocking_key_address_numeric(row.business_address)),
        ("Address token overlap", lambda row: generate_blocking_key_address_tokens(row.business_address)),
    )

    for pass_name, key_func in passes:
        if verbose:
            print(f"\nPass: {pass_name} + country")
        s1_index = defaultdict(list)
        for row in s1_df.itertuples(index=False):
            keys = key_func(row)
            if isinstance(keys, str):
                keys = (keys,) if keys else ()
            for key in keys:
                if key:
                    s1_index[(key, row.country)].append(row.entity_id)

        added = 0
        for frame in target_frames:
            for start in range(0, len(frame), target_chunk_size):
                # Only this slice is traversed; no target-side inverted index
                # and no S2+S3 concatenation are retained.
                chunk = frame.iloc[start:start + target_chunk_size]
                for row in chunk.itertuples(index=False):
                    keys = key_func(row)
                    if isinstance(keys, str):
                        keys = (keys,) if keys else ()
                    for key in keys:
                        if not key:
                            continue
                        for s1_id in s1_index.get((key, row.country), ()):
                            bucket = candidates[s1_id]
                            if row.entity_id not in bucket and (
                                max_candidates_per_entity is None
                                or len(bucket) < max_candidates_per_entity
                            ):
                                bucket.add(row.entity_id)
                                added += 1
        if verbose:
            print(f"  Added {added:,} candidate pairs; {len(candidates):,} S1 entities have candidates")

    # Ensure every S1 entity has an entry (even if empty)
    for entity_id in s1_df['entity_id']:
        candidates.setdefault(entity_id, set())

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
