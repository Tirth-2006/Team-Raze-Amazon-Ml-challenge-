"""
Fast baseline training - heavily sampled for speed
Train on 5% of data, simpler features, faster blocking
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from scorer import macro_f05


def load_sampled_data(base_path='../../../student_resource/dataset', sample_pct=5):
    """Load heavily sampled training data"""
    print(f"Loading {sample_pct}% sample of training data...")

    # Sample S1 first
    train_s1 = pd.read_csv(f'{base_path}/train/train_source1.tsv', sep='\t')
    n_sample = int(len(train_s1) * sample_pct / 100)
    train_s1 = train_s1.sample(n=n_sample, random_state=42).reset_index(drop=True)

    # Load S2 and S3 (keep full for matching)
    train_s2 = pd.read_csv(f'{base_path}/train/train_source2.tsv', sep='\t')
    train_s3 = pd.read_csv(f'{base_path}/train/train_source3.tsv', sep='\t')

    # Load ground truth for sampled S1
    train_gt = pd.read_csv(f'{base_path}/train/train_ground_truth.tsv', sep='\t')
    train_gt = train_gt[train_gt['source1_entity_id'].isin(train_s1['entity_id'])]

    print(f"S1={len(train_s1):,}, S2={len(train_s2):,}, S3={len(train_s3):,}, GT={len(train_gt):,}")
    return train_s1, train_s2, train_s3, train_gt


def simple_blocking(s1_df, s2_df, s3_df, max_candidates=50):
    """
    Simplified blocking using only name-based keys
    Limit candidates per entity for speed
    """
    from normalization import normalize_business_name

    print("Running simplified blocking...")
    candidates = defaultdict(set)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Build index: (first_token, country) -> list of target ids
    target_index = defaultdict(list)
    for _, row in target_df.iterrows():
        name_norm = normalize_business_name(row['business_name'])
        tokens = [t for t in name_norm.split() if len(t) >= 3]
        if tokens:
            first_token = tokens[0]
            key = (first_token, row['country'])
            target_index[key].append(row['entity_id'])

    # Match S1 entities
    for _, row in s1_df.iterrows():
        s1_id = row['entity_id']
        name_norm = normalize_business_name(row['business_name'])
        tokens = [t for t in name_norm.split() if len(t) >= 3]

        if tokens:
            first_token = tokens[0]
            key = (first_token, row['country'])
            if key in target_index:
                # Limit candidates
                cands = target_index[key][:max_candidates]
                candidates[s1_id].update(cands)

    # Ensure all S1 entities have an entry
    for _, row in s1_df.iterrows():
        if row['entity_id'] not in candidates:
            candidates[row['entity_id']] = set()

    total_cands = sum(len(v) for v in candidates.values())
    print(f"Generated {total_cands:,} candidate pairs for {len(candidates):,} entities")
    print(f"Avg candidates per entity: {total_cands/len(candidates):.1f}")

    return dict(candidates)


def compute_simple_features(s1_row, target_row):
    """Compute essential features only"""
    from rapidfuzz import fuzz
    from normalization import normalize_business_name, normalize_address

    s1_name = normalize_business_name(s1_row['business_name'])
    target_name = normalize_business_name(target_row['business_name'])
    s1_addr = normalize_address(s1_row['business_address'])
    target_addr = normalize_address(target_row['business_address'])

    return {
        'name_ratio': fuzz.ratio(s1_name, target_name) / 100.0,
        'name_partial': fuzz.partial_ratio(s1_name, target_name) / 100.0,
        'addr_ratio': fuzz.ratio(s1_addr, target_addr) / 100.0,
        'country_match': int(s1_row['country'] == target_row['country']),
        'is_s2': int(target_row['entity_id'].startswith('S2-')),
    }


def build_simple_features(s1_df, target_df, pairs):
    """Build feature matrix with simple features"""
    print(f"Building features for {len(pairs):,} pairs...")

    s1_idx = s1_df.set_index('entity_id')
    target_idx = target_df.set_index('entity_id')

    features = []
    for s1_id, target_id in pairs:
        s1_row = s1_idx.loc[s1_id]
        target_row = target_idx.loc[target_id]
        feat = compute_simple_features(s1_row, target_row)
        feat['s1_id'] = s1_id
        feat['target_id'] = target_id
        features.append(feat)

    return pd.DataFrame(features)


def parse_gt(train_gt):
    """Parse ground truth"""
    gt_dict = {}
    for _, row in train_gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if pd.isna(matched) or matched == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(matched.split(','))
    return gt_dict


def main():
    print("="*80)
    print("FAST BASELINE TRAINING")
    print("="*80)

    # Load 5% sample
    train_s1, train_s2, train_s3, train_gt = load_sampled_data(sample_pct=5)
    gt_dict = parse_gt(train_gt)

    # Train/val split
    train_ids, val_ids = train_test_split(train_s1['entity_id'].values, test_size=0.2, random_state=42)
    train_s1_df = train_s1[train_s1['entity_id'].isin(train_ids)]
    val_s1_df = train_s1[train_s1['entity_id'].isin(val_ids)]

    target_df = pd.concat([train_s2, train_s3], ignore_index=True)

    # Blocking
    print("\nBlocking - Train...")
    train_cands = simple_blocking(train_s1_df, train_s2, train_s3, max_candidates=50)

    print("\nBlocking - Val...")
    val_cands = simple_blocking(val_s1_df, train_s2, train_s3, max_candidates=50)

    # Generate training pairs
    print("\nGenerating training pairs...")
    train_pos, train_neg = [], []
    for s1_id in train_ids:
        if s1_id not in gt_dict:
            continue
        true_matches = gt_dict[s1_id]
        cands = train_cands.get(s1_id, set())

        for tid in cands:
            if tid in true_matches:
                train_pos.append((s1_id, tid))
            else:
                train_neg.append((s1_id, tid))

    # Sample negatives
    if len(train_neg) > len(train_pos) * 3:
        train_neg = list(np.random.choice(len(train_neg), len(train_pos) * 3, replace=False))
        train_neg = [train_neg[i] for i in range(len(train_neg))]

    print(f"Positive: {len(train_pos):,}, Negative: {len(train_neg):,}")

    all_pairs = train_pos + train_neg
    labels = [1] * len(train_pos) + [0] * len(train_neg)

    # Features
    train_feat = build_simple_features(train_s1_df, target_df, all_pairs)
    train_feat['label'] = labels

    # Val pairs
    print("\nGenerating val pairs...")
    val_pos, val_neg = [], []
    for s1_id in val_ids:
        if s1_id not in gt_dict:
            continue
        true_matches = gt_dict[s1_id]
        cands = val_cands.get(s1_id, set())

        for tid in cands:
            if tid in true_matches:
                val_pos.append((s1_id, tid))
            else:
                val_neg.append((s1_id, tid))

    print(f"Positive: {len(val_pos):,}, Negative: {len(val_neg):,}")

    val_pairs = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)

    val_feat = build_simple_features(val_s1_df, target_df, val_pairs)
    val_feat['label'] = val_labels

    # Train
    print("\nTraining XGBoost...")
    feature_cols = ['name_ratio', 'name_partial', 'addr_ratio', 'country_match', 'is_s2']

    X_train = train_feat[feature_cols].values
    y_train = train_feat['label'].values
    X_val = val_feat[feature_cols].values
    y_val = val_feat['label'].values

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.2,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)

    # Save
    os.makedirs('../../../models', exist_ok=True)
    with open('../../../models/xgb_baseline.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)

    # Threshold tuning
    print("\nThreshold tuning...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_feat['prob'] = val_probs

    val_preds = defaultdict(list)
    for _, row in val_feat.iterrows():
        val_preds[row['s1_id']].append((row['target_id'], row['prob']))

    val_gt = {k: v for k, v in gt_dict.items() if k in val_ids}

    best_thresh, best_f05 = 0.5, 0.0
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = {}
        for s1_id in val_ids:
            cands = val_preds.get(s1_id, [])
            matches = {tid for tid, prob in cands if prob >= thresh}
            y_pred[s1_id] = matches

        f05 = macro_f05(val_gt, y_pred)
        print(f"Threshold {thresh:.2f}: F0.5 = {f05:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    with open('../../../models/threshold.txt', 'w') as f:
        f.write(str(best_thresh))

    print(f"\n=== COMPLETE ===")
    print(f"Best threshold: {best_thresh}")
    print(f"Val F0.5: {best_f05:.4f}")


if __name__ == '__main__':
    main()
