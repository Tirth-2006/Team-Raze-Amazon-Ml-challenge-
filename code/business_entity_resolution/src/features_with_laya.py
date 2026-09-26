"""
Enhanced feature engineering with Laya integration stub

This adds the Laya decision model as one feature input to XGBoost.
The Laya score is computed for each candidate pair and becomes one column
in the feature matrix, alongside string similarity features.

To complete integration:
1. Install Laya: pip install laya (or whatever the actual package name is)
2. Fill in the laya_score() function below with actual model loading
3. Run Model A vs Model C ablation test to validate improvement
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from normalization import normalize_business_name, normalize_address, extract_tokens, extract_numeric_tokens


# ============================================================
# LAYA INTEGRATION — fill in with your Hugging Face model load
# ============================================================
#
# Expected checkpoint: convaiinnovations/laya  (English, 421M, Apache 2.0)
# Load once at module import time, not per-call — reloading the model
# per pair would dominate your runtime on 1.7M+ candidates.
#
# from laya import load  # or however Convai's SDK exposes it
# _laya_model = load("convaiinnovations/laya")

_LAYA_ENABLED = False  # Set to True after wiring up the model

def laya_score(record_a: dict, record_b: dict) -> float:
    """
    Ask Laya's 'noul' (yes/no) decision type whether two business
    records describe the same real-world entity.

    record_a / record_b: {"business_name": str, "business_address": str,
                           "country": str}

    Returns: calibrated P(true) as a float in [0, 1].
    Must never raise — on any load/inference failure, return a neutral
    0.5 so a bad Laya call degrades to "uninformative feature" rather
    than crashing the whole pairwise feature pipeline.
    """
    if not _LAYA_ENABLED:
        return 0.5  # Neutral score when Laya is not loaded

    try:
        # -------- TODO: replace with actual Laya inference call --------
        #
        # state = f"Record A: {record_a['business_name']}, {record_a['business_address']}, {record_a['country']}\n" \
        #         f"Record B: {record_b['business_name']}, {record_b['business_address']}, {record_b['country']}"
        # result = _laya_model.decide(
        #     state=state,
        #     question="Are these two records describing the same real-world business entity?",
        #     question_type="noul",  # yes/no decision
        # )
        # return float(result.probability)
        #
        # -----------------------------------------------------------------
        return 0.5  # Placeholder until wired up

    except Exception as e:
        # Log the error but don't crash — return neutral score
        print(f"WARNING: Laya inference failed: {e}")
        return 0.5


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_pairwise_features(s1_row: pd.Series, target_row: pd.Series, include_laya: bool = False) -> dict:
    """
    Compute similarity features for a single pair

    Features:
    - Name similarities: Jaro-Winkler, token Jaccard
    - Address similarities: Jaro-Winkler, token Jaccard, numeric overlap
    - Exact match flags: country
    - Source indicator: is_s2, is_s3
    - (Optional) Laya decision model score

    Args:
        s1_row: S1 entity row
        target_row: S2 or S3 entity row
        include_laya: If True, add laya_score feature (requires Laya model loaded)
    """
    features = {}

    # Normalize fields
    s1_name_norm = normalize_business_name(s1_row['business_name'])
    target_name_norm = normalize_business_name(target_row['business_name'])

    s1_addr_norm = normalize_address(s1_row['business_address'])
    target_addr_norm = normalize_address(target_row['business_address'])

    # === NAME FEATURES ===

    # Jaro-Winkler on normalized names
    features['name_jaro_winkler'] = fuzz.ratio(s1_name_norm, target_name_norm) / 100.0

    # Token Jaccard on names
    s1_name_tokens = extract_tokens(s1_name_norm)
    target_name_tokens = extract_tokens(target_name_norm)
    features['name_token_jaccard'] = jaccard_similarity(s1_name_tokens, target_name_tokens)

    # Partial ratio (handles substring matches)
    features['name_partial_ratio'] = fuzz.partial_ratio(s1_name_norm, target_name_norm) / 100.0

    # === ADDRESS FEATURES ===

    # Jaro-Winkler on normalized addresses
    features['addr_jaro_winkler'] = fuzz.ratio(s1_addr_norm, target_addr_norm) / 100.0

    # Token Jaccard on addresses
    s1_addr_tokens = extract_tokens(s1_addr_norm)
    target_addr_tokens = extract_tokens(target_addr_norm)
    features['addr_token_jaccard'] = jaccard_similarity(s1_addr_tokens, target_addr_tokens)

    # Numeric overlap in addresses (street numbers, zip codes)
    s1_addr_nums = extract_numeric_tokens(s1_addr_norm)
    target_addr_nums = extract_numeric_tokens(target_addr_norm)
    features['addr_numeric_jaccard'] = jaccard_similarity(s1_addr_nums, target_addr_nums)

    # Partial ratio for addresses
    features['addr_partial_ratio'] = fuzz.partial_ratio(s1_addr_norm, target_addr_norm) / 100.0

    # === EXACT MATCH FLAGS ===

    features['country_match'] = int(s1_row['country'] == target_row['country'])

    # === SOURCE INDICATOR ===

    features['is_s2'] = int(target_row['entity_id'].startswith('S2-'))
    features['is_s3'] = int(target_row['entity_id'].startswith('S3-'))

    # === COMBINED FEATURES ===

    # Weighted average of name and address similarities
    features['combined_avg'] = 0.6 * features['name_jaro_winkler'] + 0.4 * features['addr_jaro_winkler']

    # Min similarity (conservative - both must match)
    features['combined_min'] = min(features['name_jaro_winkler'], features['addr_jaro_winkler'])

    # Token overlap product
    features['token_product'] = features['name_token_jaccard'] * features['addr_token_jaccard']

    # === LAYA DECISION MODEL (optional) ===

    if include_laya and _LAYA_ENABLED:
        record_a = {
            'business_name': s1_row['business_name'],
            'business_address': s1_row['business_address'],
            'country': s1_row['country']
        }
        record_b = {
            'business_name': target_row['business_name'],
            'business_address': target_row['business_address'],
            'country': target_row['country']
        }
        features['laya_score'] = laya_score(record_a, record_b)

    return features


def build_feature_matrix(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_pairs: list,
    include_laya: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build feature matrix for candidate pairs

    Args:
        s1_df: Source 1 DataFrame
        target_df: Source 2 + Source 3 combined DataFrame
        candidate_pairs: List of (s1_id, target_id) tuples
        include_laya: If True, compute Laya scores (requires model loaded)
        verbose: Print progress

    Returns:
        DataFrame with features for each pair
    """
    if verbose:
        print(f"Computing features for {len(candidate_pairs):,} candidate pairs...")
        if include_laya:
            if _LAYA_ENABLED:
                print("  Laya decision model: ENABLED")
            else:
                print("  WARNING: include_laya=True but Laya not loaded (_LAYA_ENABLED=False)")

    # Index DataFrames by entity_id for fast lookup
    s1_indexed = s1_df.set_index('entity_id')
    target_indexed = target_df.set_index('entity_id')

    feature_rows = []

    for i, (s1_id, target_id) in enumerate(candidate_pairs):
        if verbose and (i + 1) % 100000 == 0:
            print(f"  Processed {i+1:,} / {len(candidate_pairs):,} pairs...")

        s1_row = s1_indexed.loc[s1_id]
        target_row = target_indexed.loc[target_id]

        features = compute_pairwise_features(s1_row, target_row, include_laya=include_laya)
        features['s1_id'] = s1_id
        features['target_id'] = target_id

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    if verbose:
        print(f"Feature matrix built: {len(feature_df):,} rows x {len(feature_df.columns)} columns")
        print(f"Feature columns: {list(feature_df.columns)}")

    return feature_df


def add_tfidf_features(
    feature_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add TF-IDF cosine similarity features (optional - computationally expensive)

    Args:
        feature_df: Existing feature DataFrame
        s1_df: Source 1 DataFrame
        target_df: Source 2 + Source 3 DataFrame
        verbose: Print progress

    Returns:
        feature_df with added TF-IDF features
    """
    if verbose:
        print("Computing TF-IDF features (this may take a while)...")

    # Index DataFrames
    s1_indexed = s1_df.set_index('entity_id')
    target_indexed = target_df.set_index('entity_id')

    # Get unique entities in feature_df
    unique_s1_ids = feature_df['s1_id'].unique()
    unique_target_ids = feature_df['target_id'].unique()

    # Build normalized text corpus
    s1_texts = []
    s1_ids_ordered = []
    for s1_id in unique_s1_ids:
        row = s1_indexed.loc[s1_id]
        text = f"{normalize_business_name(row['business_name'])} {normalize_address(row['business_address'])}"
        s1_texts.append(text)
        s1_ids_ordered.append(s1_id)

    target_texts = []
    target_ids_ordered = []
    for target_id in unique_target_ids:
        row = target_indexed.loc[target_id]
        text = f"{normalize_business_name(row['business_name'])} {normalize_address(row['business_address'])}"
        target_texts.append(text)
        target_ids_ordered.append(target_id)

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    all_texts = s1_texts + target_texts
    vectorizer.fit(all_texts)

    # Transform
    s1_tfidf = vectorizer.transform(s1_texts)
    target_tfidf = vectorizer.transform(target_texts)

    # Build lookup dictionaries for TF-IDF vectors
    s1_tfidf_dict = {s1_id: s1_tfidf[i] for i, s1_id in enumerate(s1_ids_ordered)}
    target_tfidf_dict = {target_id: target_tfidf[i] for i, target_id in enumerate(target_ids_ordered)}

    # Compute cosine similarity for each pair
    tfidf_cosines = []
    for _, row in feature_df.iterrows():
        s1_vec = s1_tfidf_dict[row['s1_id']]
        target_vec = target_tfidf_dict[row['target_id']]
        cosine = cosine_similarity(s1_vec, target_vec)[0, 0]
        tfidf_cosines.append(cosine)

    feature_df['tfidf_cosine'] = tfidf_cosines

    if verbose:
        print(f"TF-IDF feature added. Mean cosine: {np.mean(tfidf_cosines):.3f}")

    return feature_df
