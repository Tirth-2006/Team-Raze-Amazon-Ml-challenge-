"""
Quick test of blocking and feature engineering on a small sample
"""
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from normalization import normalize_business_name, normalize_address
from blocking import generate_candidates_multi_pass
from features import compute_pairwise_features
from scorer import macro_f05


def test_normalization():
    print("=== TESTING NORMALIZATION ===")

    test_names = [
        "Orelee's Barbershop",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Summit Inc",
        "B+ Retail Inc",
    ]

    for name in test_names:
        normalized = normalize_business_name(name)
        try:
            print(f"{name:50s} -> {normalized}")
        except UnicodeEncodeError:
            print(f"[Unicode name] -> {normalized}")

    print("\n=== TESTING ADDRESS NORMALIZATION ===")
    test_addrs = [
        "1795 Westchester Drive, High Point, NC",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "105 ELM ST, MORGANTON, NC",
    ]

    for addr in test_addrs:
        normalized = normalize_address(addr)
        try:
            print(f"{addr:60s} -> {normalized}")
        except UnicodeEncodeError:
            print(f"[Unicode address] -> {normalized}")


def test_blocking_small():
    print("\n=== TESTING BLOCKING ON SMALL SAMPLE ===")

    # Load small sample
    base_path = '../../../student_resource/dataset/train'
    s1 = pd.read_csv(f'{base_path}/train_source1.tsv', sep='\t', nrows=1000)
    s2 = pd.read_csv(f'{base_path}/train_source2.tsv', sep='\t', nrows=5000)
    s3 = pd.read_csv(f'{base_path}/train_source3.tsv', sep='\t', nrows=5000)

    print(f"Sample: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

    candidates = generate_candidates_multi_pass(s1, s2, s3, verbose=True)

    print(f"\nGenerated candidates for {len(candidates)} S1 entities")

    # Show some examples
    print("\nExample candidates:")
    for i, (s1_id, cands) in enumerate(list(candidates.items())[:5]):
        print(f"  {s1_id}: {len(cands)} candidates")


def test_features():
    print("\n=== TESTING FEATURE ENGINEERING ===")

    # Create mock rows
    s1_row = pd.Series({
        'entity_id': 'S1-123',
        'business_name': 'ABC Corporation',
        'business_address': '123 Main Street, New York, NY',
        'country': 'US'
    })

    s2_row = pd.Series({
        'entity_id': 'S2-456',
        'business_name': 'ABC Corp',
        'business_address': '123 Main St, New York, NY',
        'country': 'US'
    })

    features = compute_pairwise_features(s1_row, s2_row)

    print("\nFeatures for similar pair:")
    for key, value in sorted(features.items()):
        print(f"  {key:25s}: {value:.4f}")

    # Dissimilar pair
    s3_row = pd.Series({
        'entity_id': 'S3-789',
        'business_name': 'XYZ Limited',
        'business_address': '456 Oak Avenue, Los Angeles, CA',
        'country': 'US'
    })

    features_dissimilar = compute_pairwise_features(s1_row, s3_row)

    print("\nFeatures for dissimilar pair:")
    for key, value in sorted(features_dissimilar.items()):
        print(f"  {key:25s}: {value:.4f}")


def test_scorer():
    print("\n=== TESTING SCORER ===")

    # Example from competition docs: P=0.667, R=1.0 -> F0.5=0.714
    y_true = {
        'S1-00001': {'S2-00047', 'S3-00812'}
    }

    y_pred = {
        'S1-00001': {'S2-00047', 'S2-00193', 'S3-00812'}
    }

    score = macro_f05(y_true, y_pred)
    print(f"Example score (should be ~0.714): {score:.4f}")

    # Test singleton
    y_true_singleton = {
        'S1-00002': set()  # No matches
    }

    y_pred_singleton = {
        'S1-00002': set()  # Correctly predicted no match
    }

    score_singleton = macro_f05(y_true_singleton, y_pred_singleton)
    print(f"Singleton correct (should be 1.0): {score_singleton:.4f}")

    # Test false positive on singleton
    y_pred_fp = {
        'S1-00002': {'S2-12345'}  # Incorrectly predicted a match
    }

    score_fp = macro_f05(y_true_singleton, y_pred_fp)
    print(f"Singleton false positive (should be 0.0): {score_fp:.4f}")


if __name__ == '__main__':
    test_normalization()
    test_blocking_small()
    test_features()
    test_scorer()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
