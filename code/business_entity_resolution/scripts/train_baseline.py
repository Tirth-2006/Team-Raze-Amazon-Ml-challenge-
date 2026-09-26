"""
Main training pipeline - Stage 1: Build baseline with entity-level train/val split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os
import sys
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from normalization import normalize_business_name, normalize_address
from blocking import generate_candidates_multi_pass
from features import build_feature_matrix, compute_pairwise_features
from scorer import macro_f05


def load_data(base_path='../../student_resource/dataset'):
    """Load training data"""
    print("Loading training data...")

    train_s1 = pd.read_csv(f'{base_path}/train/train_source1.tsv', sep='\t')
    train_s2 = pd.read_csv(f'{base_path}/train/train_source2.tsv', sep='\t')
    train_s3 = pd.read_csv(f'{base_path}/train/train_source3.tsv', sep='\t')
    train_gt = pd.read_csv(f'{base_path}/train/train_ground_truth.tsv', sep='\t')

    print(f"Loaded: S1={len(train_s1):,}, S2={len(train_s2):,}, S3={len(train_s3):,}, GT={len(train_gt):,}")

    return train_s1, train_s2, train_s3, train_gt


def parse_ground_truth(train_gt: pd.DataFrame) -> dict:
    """Parse ground truth into dict: s1_id -> set of matched ids"""
    gt_dict = {}

    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = row['matched_entity_ids']

        if pd.isna(matched_str) or matched_str == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(matched_str.split(','))

    return gt_dict


def entity_level_split(train_s1: pd.DataFrame, train_gt: pd.DataFrame, test_size=0.1, random_state=42):
    """
    Split S1 entities into train/val - CRITICAL to avoid leakage

    Returns:
        train_s1_ids, val_s1_ids
    """
    print(f"\nCreating entity-level train/val split (test_size={test_size})...")

    all_s1_ids = train_s1['entity_id'].values
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Train entities: {len(train_s1_ids):,}")
    print(f"Val entities: {len(val_s1_ids):,}")

    return set(train_s1_ids), set(val_s1_ids)


def generate_training_pairs(
    s1_ids: set,
    candidates: dict,
    gt_dict: dict,
    target_df: pd.DataFrame
) -> tuple:
    """
    Generate labeled training pairs from candidates

    Returns:
        positive_pairs: list of (s1_id, target_id) true matches
        negative_pairs: list of (s1_id, target_id) non-matches
    """
    positive_pairs = []
    negative_pairs = []

    for s1_id in s1_ids:
        if s1_id not in gt_dict:
            continue

        true_matches = gt_dict[s1_id]
        candidate_set = candidates.get(s1_id, set())

        # Positive pairs: candidates that are true matches
        for target_id in candidate_set:
            if target_id in true_matches:
                positive_pairs.append((s1_id, target_id))

        # Negative pairs: candidates that are NOT true matches
        for target_id in candidate_set:
            if target_id not in true_matches:
                negative_pairs.append((s1_id, target_id))

    return positive_pairs, negative_pairs


def compute_blocking_recall(candidates: dict, gt_dict: dict) -> dict:
    """
    Compute blocking recall: what fraction of true matches were captured
    """
    entity_recalls = []
    total_true_matches = 0
    total_found = 0

    for s1_id, true_matches in gt_dict.items():
        if not true_matches:  # Skip singletons
            continue

        candidate_set = candidates.get(s1_id, set())
        found = len(true_matches & candidate_set)

        entity_recalls.append(found / len(true_matches))
        total_true_matches += len(true_matches)
        total_found += found

    macro_recall = np.mean(entity_recalls) if entity_recalls else 0.0
    micro_recall = total_found / total_true_matches if total_true_matches > 0 else 0.0

    return {
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
        'total_true_matches': total_true_matches,
        'total_found': total_found
    }


def main():
    print("="*80)
    print("STAGE 1: BASELINE PIPELINE - ENTITY-LEVEL TRAIN/VAL SPLIT")
    print("="*80)

    # Load data
    train_s1, train_s2, train_s3, train_gt = load_data()
    gt_dict = parse_ground_truth(train_gt)

    # Entity-level split
    train_s1_ids, val_s1_ids = entity_level_split(train_s1, train_gt, test_size=0.1)

    # Split S1 dataframe
    train_s1_df = train_s1[train_s1['entity_id'].isin(train_s1_ids)].copy()
    val_s1_df = train_s1[train_s1['entity_id'].isin(val_s1_ids)].copy()

    print(f"\nTrain S1 entities: {len(train_s1_df):,}")
    print(f"Val S1 entities: {len(val_s1_df):,}")

    # Combine S2 and S3
    target_df = pd.concat([train_s2, train_s3], ignore_index=True)
    print(f"Target pool (S2+S3): {len(target_df):,}")

    # === BLOCKING - TRAIN SET ===
    print("\n" + "="*80)
    print("BLOCKING - TRAIN SET")
    print("="*80)
    train_candidates = generate_candidates_multi_pass(train_s1_df, train_s2, train_s3, verbose=True)

    # Measure blocking recall on train
    print("\n=== BLOCKING RECALL (TRAIN) ===")
    train_gt_dict = {k: v for k, v in gt_dict.items() if k in train_s1_ids}
    train_blocking_stats = compute_blocking_recall(train_candidates, train_gt_dict)
    print(f"Macro recall: {train_blocking_stats['macro_recall']:.4f}")
    print(f"Micro recall: {train_blocking_stats['micro_recall']:.4f}")
    print(f"Found {train_blocking_stats['total_found']:,} / {train_blocking_stats['total_true_matches']:,} true matches")

    # === BLOCKING - VAL SET ===
    print("\n" + "="*80)
    print("BLOCKING - VALIDATION SET")
    print("="*80)
    val_candidates = generate_candidates_multi_pass(val_s1_df, train_s2, train_s3, verbose=True)

    # Measure blocking recall on val
    print("\n=== BLOCKING RECALL (VAL) ===")
    val_gt_dict = {k: v for k, v in gt_dict.items() if k in val_s1_ids}
    val_blocking_stats = compute_blocking_recall(val_candidates, val_gt_dict)
    print(f"Macro recall: {val_blocking_stats['macro_recall']:.4f}")
    print(f"Micro recall: {val_blocking_stats['micro_recall']:.4f}")
    print(f"Found {val_blocking_stats['total_found']:,} / {val_blocking_stats['total_true_matches']:,} true matches")

    # === FEATURE ENGINEERING - TRAIN ===
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - TRAIN SET")
    print("="*80)

    train_pos_pairs, train_neg_pairs = generate_training_pairs(
        train_s1_ids, train_candidates, gt_dict, target_df
    )

    print(f"Positive pairs: {len(train_pos_pairs):,}")
    print(f"Negative pairs: {len(train_neg_pairs):,}")
    print(f"Class ratio (pos:neg): 1:{len(train_neg_pairs)/len(train_pos_pairs):.1f}")

    # Sample negatives if too many
    max_negatives = len(train_pos_pairs) * 5  # 1:5 ratio
    if len(train_neg_pairs) > max_negatives:
        print(f"Sampling {max_negatives:,} negatives from {len(train_neg_pairs):,}...")
        np.random.seed(42)
        train_neg_pairs = list(np.random.choice(
            len(train_neg_pairs),
            size=max_negatives,
            replace=False
        ))
        train_neg_pairs = [train_neg_pairs[i] for i in range(len(train_neg_pairs))]

    all_train_pairs = train_pos_pairs + train_neg_pairs
    train_labels = [1] * len(train_pos_pairs) + [0] * len(train_neg_pairs)

    print(f"Building feature matrix for {len(all_train_pairs):,} pairs...")
    train_feature_df = build_feature_matrix(train_s1_df, target_df, all_train_pairs, verbose=True)
    train_feature_df['label'] = train_labels

    # === FEATURE ENGINEERING - VAL ===
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - VAL SET")
    print("="*80)

    val_pos_pairs, val_neg_pairs = generate_training_pairs(
        val_s1_ids, val_candidates, gt_dict, target_df
    )

    print(f"Positive pairs: {len(val_pos_pairs):,}")
    print(f"Negative pairs: {len(val_neg_pairs):,}")

    all_val_pairs = val_pos_pairs + val_neg_pairs
    val_labels = [1] * len(val_pos_pairs) + [0] * len(val_neg_pairs)

    print(f"Building feature matrix for {len(all_val_pairs):,} pairs...")
    val_feature_df = build_feature_matrix(val_s1_df, target_df, all_val_pairs, verbose=True)
    val_feature_df['label'] = val_labels

    # === TRAIN XGBOOST ===
    print("\n" + "="*80)
    print("TRAINING XGBOOST CLASSIFIER")
    print("="*80)

    feature_cols = [col for col in train_feature_df.columns if col not in ['s1_id', 'target_id', 'label']]
    print(f"Feature columns: {feature_cols}")

    X_train = train_feature_df[feature_cols].values
    y_train = train_feature_df['label'].values

    X_val = val_feature_df[feature_cols].values
    y_val = val_feature_df['label'].values

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Save model
    os.makedirs('../../models', exist_ok=True)
    with open('../../models/xgb_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to ../../models/xgb_baseline.pkl")

    # Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True):
        print(f"{feat:30s}: {imp:.4f}")

    # === THRESHOLD TUNING ON VALIDATION ===
    print("\n" + "="*80)
    print("THRESHOLD TUNING ON VALIDATION SET")
    print("="*80)

    # Predict on validation candidates
    val_probs = model.predict_proba(X_val)[:, 1]
    val_feature_df['prob'] = val_probs

    # Group by S1 entity
    val_predictions_by_entity = defaultdict(list)
    for _, row in val_feature_df.iterrows():
        val_predictions_by_entity[row['s1_id']].append((row['target_id'], row['prob']))

    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_threshold = 0.5
    best_f05 = 0.0

    for thresh in thresholds:
        y_pred = {}
        for s1_id in val_s1_ids:
            candidates = val_predictions_by_entity.get(s1_id, [])
            matches = {target_id for target_id, prob in candidates if prob >= thresh}
            y_pred[s1_id] = matches

        f05 = macro_f05(val_gt_dict, y_pred)
        print(f"Threshold {thresh:.2f}: F0.5 = {f05:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh

    print(f"\nBest threshold: {best_threshold:.2f} with F0.5 = {best_f05:.4f}")

    # Save threshold
    with open('../../models/threshold.txt', 'w') as f:
        f.write(str(best_threshold))

    print("\n" + "="*80)
    print("STAGE 1 COMPLETE - BASELINE PIPELINE READY")
    print("="*80)
    print(f"Blocking recall (val): {val_blocking_stats['macro_recall']:.4f}")
    print(f"Macro F0.5 (val): {best_f05:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")


if __name__ == '__main__':
    main()
