"""
Main training pipeline - Baseline with proper entity-level split
Training on 10% of data for speed, then full inference on test set
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from normalization import normalize_business_name, normalize_address
from blocking import generate_candidates_multi_pass
from features import build_feature_matrix, compute_pairwise_features
from scorer import macro_f05
from paths import DATASET_ROOT, MODEL_ROOT


def load_data(base_path=DATASET_ROOT):
    """Load training data"""
    print("Loading training data...")
    base_path = os.fspath(base_path)
    train_s1 = pd.read_csv(os.path.join(base_path, 'train', 'train_source1.tsv'), sep='\t')
    train_s2 = pd.read_csv(os.path.join(base_path, 'train', 'train_source2.tsv'), sep='\t')
    train_s3 = pd.read_csv(os.path.join(base_path, 'train', 'train_source3.tsv'), sep='\t')
    train_gt = pd.read_csv(os.path.join(base_path, 'train', 'train_ground_truth.tsv'), sep='\t')
    print(f"Loaded: S1={len(train_s1):,}, S2={len(train_s2):,}, S3={len(train_s3):,}, GT={len(train_gt):,}")
    return train_s1, train_s2, train_s3, train_gt


def parse_ground_truth(train_gt: pd.DataFrame) -> dict:
    """Parse ground truth into dict"""
    gt_dict = {}
    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = row['matched_entity_ids']
        if pd.isna(matched_str) or matched_str == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(matched_str.split(','))
    return gt_dict


def sample_training_data(train_s1, train_gt, sample_fraction=0.1, random_state=42):
    """Sample a fraction of training data"""
    print(f"\nSampling {sample_fraction*100:.0f}% of training data...")
    sampled_s1_ids = train_s1['entity_id'].sample(frac=sample_fraction, random_state=random_state)
    sampled_s1 = train_s1[train_s1['entity_id'].isin(sampled_s1_ids)]
    sampled_gt = train_gt[train_gt['source1_entity_id'].isin(sampled_s1_ids)]
    print(f"Sampled S1: {len(sampled_s1):,}, GT: {len(sampled_gt):,}")
    return sampled_s1, sampled_gt


def entity_level_split(s1_df, test_size=0.2, random_state=42):
    """Entity-level train/val split"""
    print(f"\nEntity-level train/val split (test_size={test_size})...")
    all_s1_ids = s1_df['entity_id'].values
    train_ids, val_ids = train_test_split(all_s1_ids, test_size=test_size, random_state=random_state)
    print(f"Train: {len(train_ids):,}, Val: {len(val_ids):,}")
    return set(train_ids), set(val_ids)


def generate_training_pairs(s1_ids, candidates, gt_dict, max_neg_ratio=3):
    """Generate labeled pairs while reservoir-sampling negatives.

    Candidate sets are bounded by the blocker, but can still be large enough
    that retaining every negative pair briefly doubles peak memory.
    """
    pos_pairs = []

    for s1_id in s1_ids:
        if s1_id not in gt_dict:
            continue
        true_matches = gt_dict[s1_id]
        candidate_set = candidates.get(s1_id, set())

        for target_id in candidate_set:
            if target_id in true_matches:
                pos_pairs.append((s1_id, target_id))

    # Reservoir sample negatives in a second pass, without a giant temporary
    # list or NumPy index array.
    negative_limit = len(pos_pairs) * max_neg_ratio
    neg_pairs = []
    seen_negatives = 0
    rng = random.Random(42)
    for s1_id in s1_ids:
        true_matches = gt_dict.get(s1_id)
        if true_matches is None:
            continue
        for target_id in candidates.get(s1_id, set()):
            if target_id in true_matches:
                continue
            seen_negatives += 1
            if len(neg_pairs) < negative_limit:
                neg_pairs.append((s1_id, target_id))
            elif negative_limit and rng.randrange(seen_negatives) < negative_limit:
                neg_pairs[rng.randrange(negative_limit)] = (s1_id, target_id)

    if seen_negatives > negative_limit:
        print(f"  Sampling negatives: {negative_limit:,} from {seen_negatives:,}")

    return pos_pairs, neg_pairs


def compute_blocking_recall(candidates, gt_dict):
    """Compute blocking recall"""
    entity_recalls = []
    total_true = 0
    total_found = 0

    for s1_id, true_matches in gt_dict.items():
        if not true_matches:
            continue
        candidate_set = candidates.get(s1_id, set())
        found = len(true_matches & candidate_set)
        entity_recalls.append(found / len(true_matches))
        total_true += len(true_matches)
        total_found += found

    macro_recall = np.mean(entity_recalls) if entity_recalls else 0.0
    micro_recall = total_found / total_true if total_true > 0 else 0.0

    return {'macro': macro_recall, 'micro': micro_recall, 'found': total_found, 'total': total_true}


def main():
    print("="*80)
    print("BASELINE TRAINING PIPELINE")
    print("="*80)

    # Load full data
    train_s1_full, train_s2, train_s3, train_gt_full = load_data()
    # Sample 10% for faster training
    train_s1, train_gt = sample_training_data(train_s1_full, train_gt_full, sample_fraction=0.1)
    gt_dict = parse_ground_truth(train_gt)

    # Entity-level split
    train_ids, val_ids = entity_level_split(train_s1, test_size=0.2)
    train_s1_df = train_s1[train_s1['entity_id'].isin(train_ids)].copy()
    val_s1_df = train_s1[train_s1['entity_id'].isin(val_ids)].copy()

    # === BLOCKING ===
    print("\n" + "="*80)
    print("BLOCKING - TRAIN")
    print("="*80)
    train_candidates = generate_candidates_multi_pass(
        train_s1_df, train_s2, train_s3, verbose=True,
        target_chunk_size=100_000, max_candidates_per_entity=2_000
    )
    train_gt_dict = {k: v for k, v in gt_dict.items() if k in train_ids}
    train_recall = compute_blocking_recall(train_candidates, train_gt_dict)
    print(f"\nBlocking recall (train): macro={train_recall['macro']:.4f}, micro={train_recall['micro']:.4f}")

    print("\n" + "="*80)
    print("BLOCKING - VAL")
    print("="*80)
    val_candidates = generate_candidates_multi_pass(
        val_s1_df, train_s2, train_s3, verbose=True,
        target_chunk_size=100_000, max_candidates_per_entity=2_000
    )
    val_gt_dict = {k: v for k, v in gt_dict.items() if k in val_ids}
    val_recall = compute_blocking_recall(val_candidates, val_gt_dict)
    print(f"\nBlocking recall (val): macro={val_recall['macro']:.4f}, micro={val_recall['micro']:.4f}")

    # Keep only target rows that survived blocking.  The previous full
    # S2+S3 concat duplicated the multi-million-row target pool in memory.
    needed_target_ids = {
        target_id
        for candidate_map in (train_candidates, val_candidates)
        for values in candidate_map.values()
        for target_id in values
    }
    target_df = pd.concat(
        [
            frame[frame['entity_id'].isin(needed_target_ids)]
            for frame in (train_s2, train_s3)
        ],
        ignore_index=True,
    )
    print(f"Target rows retained for features: {len(target_df):,}")

    # === FEATURE ENGINEERING ===
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - TRAIN")
    print("="*80)
    train_pos, train_neg = generate_training_pairs(train_ids, train_candidates, gt_dict, max_neg_ratio=3)
    print(f"Positive: {len(train_pos):,}, Negative: {len(train_neg):,}")

    all_train_pairs = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    print(f"Building features for {len(all_train_pairs):,} pairs...")

    train_feature_df = build_feature_matrix(train_s1_df, target_df, all_train_pairs, verbose=True)
    train_feature_df['label'] = train_labels

    print("\n" + "="*80)
    print("FEATURE ENGINEERING - VAL")
    print("="*80)
    val_pos, val_neg = generate_training_pairs(val_ids, val_candidates, gt_dict, max_neg_ratio=3)
    print(f"Positive: {len(val_pos):,}, Negative: {len(val_neg):,}")

    all_val_pairs = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)
    print(f"Building features for {len(all_val_pairs):,} pairs...")

    val_feature_df = build_feature_matrix(val_s1_df, target_df, all_val_pairs, verbose=True)
    val_feature_df['label'] = val_labels

    # === TRAIN MODEL ===
    print("\n" + "="*80)
    print("TRAINING XGBOOST")
    print("="*80)

    feature_cols = [col for col in train_feature_df.columns if col not in ['s1_id', 'target_id', 'label']]
    print(f"Features: {feature_cols}")

    X_train = train_feature_df[feature_cols].values
    y_train = train_feature_df['label'].values
    X_val = val_feature_df[feature_cols].values
    y_val = val_feature_df['label'].values

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    print("Training...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)

    # Save
    os.makedirs(MODEL_ROOT, exist_ok=True)
    with open(MODEL_ROOT / 'xgb_baseline.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)
    print("\nModel saved")

    # Feature importance
    print("\n=== TOP FEATURES ===")
    importance = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importance[:10]:
        print(f"{feat:25s}: {imp:.4f}")

    # === THRESHOLD TUNING ===
    print("\n" + "="*80)
    print("THRESHOLD TUNING")
    print("="*80)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_feature_df['prob'] = val_probs

    val_preds_by_entity = defaultdict(list)
    for _, row in val_feature_df.iterrows():
        val_preds_by_entity[row['s1_id']].append((row['target_id'], row['prob']))

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_threshold = 0.5
    best_f05 = 0.0

    for thresh in thresholds:
        y_pred = {}
        for s1_id in val_ids:
            cands = val_preds_by_entity.get(s1_id, [])
            matches = {tid for tid, prob in cands if prob >= thresh}
            y_pred[s1_id] = matches

        f05 = macro_f05(val_gt_dict, y_pred)
        print(f"Threshold {thresh:.2f}: F0.5 = {f05:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh

    with open(MODEL_ROOT / 'threshold.txt', 'w') as f:
        f.write(str(best_threshold))

    print(f"\n=== FINAL RESULTS ===")
    print(f"Best threshold: {best_threshold}")
    print(f"Validation F0.5: {best_f05:.4f}")
    print(f"Blocking recall (val): {val_recall['macro']:.4f}")

    print("\n" + "="*80)
    print("BASELINE COMPLETE - Ready for test inference")
    print("="*80)


if __name__ == '__main__':
    main()
