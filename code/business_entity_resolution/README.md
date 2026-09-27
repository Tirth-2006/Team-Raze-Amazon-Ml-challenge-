# Business Entity Resolution - Amazon ML Challenge 2026

## Team: Raze

## Solution Overview

This solution tackles the business entity resolution problem using a two-stage pipeline:
1. **Blocking/Candidate Generation**: Multi-pass blocking with name and address-based keys
2. **Matching**: Similarity-based classification to predict final matches

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Create a Memory-Safe Baseline Submission
```bash
cd scripts
python memory_efficient_submission_v3_ultra.py
```

This generates:
- `output/candidate_pairs.tsv` - blocking results
- `output/matching_results.tsv` - final predictions

### Validate Submission
```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```

## Approach

### Stage 1: Blocking (Candidate Generation)

**Goal**: Reduce the comparison space from O(n²) to manageable size while maintaining high recall.

**Strategy**: Multi-pass blocking with complementary keys
- Pass 1: Name prefix (first 4 chars) + country
- Pass 2: Shared name tokens + country  
- Pass 3: Address number + country
- Pass 4: Address token overlap + country

**Key Design Decisions**:
- Country is ALWAYS part of the blocking key (matches cannot cross countries)
- Conservative normalization to handle unseen countries (France in test but not train)
- Union of multiple passes to maximize recall
- Track candidates-per-entity as it directly affects ranking
- Target records are scanned in bounded chunks; the blocker does not build a
  target-side inverted index or concatenate S2 and S3. The fast training entry
  point also caps each S1 candidate set at 2,000 records to keep the
  training/validation feature matrices bounded on multi-million-row targets.

### Stage 2: Matching

**Features**:
- Name similarity: Jaro-Winkler, token Jaccard, partial ratio
- Address similarity: Jaro-Winkler, token Jaccard, numeric overlap
- Exact match flags: country
- Source indicator: is_S2, is_S3
- Combined features: weighted average, min, product

**Model**: XGBoost classifier
- Handles class imbalance via `scale_pos_weight`
- Separate thresholds for S2 and S3 matches (tuned on validation)
- Explicit singleton handling (predict empty if no candidate exceeds threshold)

**Evaluation**: Macro F0.5 (precision-weighted)
- Computed per-entity then averaged
- Singletons contribute 1.0 when correctly predicted empty, 0.0 otherwise

### Normalization

**Conservative approach** given open country set:
- Transliteration: `unidecode` to handle Hindi/Kannada/French
- Lowercase, remove punctuation
- Normalize common legal suffixes (Corp, Inc, Ltd, Pvt)
- Normalize street abbreviations (St, Rd, Ave, Dr)
- Preserve original data alongside normalized versions

### Handling France (Unseen Country)

France appears in 15% of test S1 entities but has ZERO training examples.

**Mitigation**:
- No hard-coded country filters or one-hot encoding
- All normalization is language-agnostic (via unidecode)
- Blocking keys work identically for all countries
- Model sees country only as exact-match flag, not as categorical feature

## Repository Structure

```
code/business_entity_resolution/
├── src/
│   ├── normalization.py      # Text normalization
│   ├── blocking.py            # Multi-pass blocking
│   ├── features.py            # Pairwise feature engineering
│   ├── scorer.py              # Official F0.5 scorer
│   └── __init__.py
├── scripts/
│   ├── ultra_fast_baseline.py     # Fast rule-based baseline
│   ├── train_baseline_fast.py     # ML training pipeline
│   ├── predict_test.py            # Test set inference
│   └── test_components.py         # Unit tests
├── requirements.txt
└── README.md
```

## Reproducibility

### Option 1: Memory-safe rule-based baseline
```bash
cd scripts
python memory_efficient_submission_v3_ultra.py
```
This version streams Source 2 and Source 3 in chunks and writes one row for
every Source 1 test entity. Runtime depends on disk speed and available CPU.

### Option 1 (recommended): Disk-backed streaming submission
```bash
python scripts/streaming_submission.py --rebuild-index
```
This builds a local SQLite index from the two target files once, then streams
Source 1 in bounded chunks. It keeps only the current Source 1 chunk and the
current candidate scores in memory, and writes both required output files
incrementally. The candidates are ranked and capped immediately before the
matching thresholds are applied, so `candidate_pairs.tsv` is the exact final
candidate set fed to the matcher. The index is stored at
`output/targets.sqlite`. The index uses composite blocking keys: significant
name tokens, exact normalized names, and address number plus address token.
Reuse it on subsequent runs without `--rebuild-index`; an older index must be
rebuilt once.

### Option 2: ML Pipeline (Better performance, slower)
```bash
# Train model
cd scripts  
python train_baseline_fast.py  # Trains on 10% sample

# Generate predictions
python predict_test.py
```

## Performance Characteristics

**Blocking**:
- Recall: ~95% (measured on validation set)
- Avg candidates per entity: ~1000-1500
- This sets the ceiling for final matching performance

**Matching** (on validation set):
- Precision: ~0.75-0.85
- Recall: ~0.70-0.80  
- Macro F0.5: ~0.75-0.80 (precision-weighted, as required)

## Key Challenges

1. **Scale**: 2.2M training entities, 1.7M test entities
2. **Noisy data**: Abbreviations, typos, transliterations, missing fields
3. **Unseen country**: France in test, not in train
4. **Class imbalance**: ~5% of candidates are true matches
5. **Macro F0.5**: Per-entity scoring means singletons matter
6. **Candidate set size**: Directly affects ranking (smaller is better)

## Future Improvements

Given more time, we would explore:
1. **Laya embeddings** as additional features (license-compliant, <1B params)
2. **TF-IDF cosine similarity** on combined name+address text  
3. **Probability calibration** (Platt scaling) on XGBoost scores
4. **Source-aware threshold tuning** (separate thresholds for S2 vs S3)
5. **Cross-source consistency** (if S1→S2 and S1→S3, check S2↔S3 compatibility)
6. **Conflict resolution** (if same S2/S3 claimed by multiple S1, pick highest confidence)
7. **Ensemble methods** (combine multiple blocking strategies, multiple models)

## License

Model: XGBoost (Apache 2.0)
All dependencies: MIT or Apache 2.0 compatible
