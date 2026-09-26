# Pipeline Fixes - Critical Issues Resolved

**Date:** 2026-09-26  
**Status:** Ready for validation testing

---

## Summary

This document tracks the 5 critical issues identified in the entity matching pipeline and their fixes. Each issue could cause silent failures at scale, affecting the final F0.5 score without obvious error messages.

---

## Issue #1: Two Different Blocking Implementations ❌ → ✅

### Problem
- **`blocking.py`** (the tested module): 4-pass blocking with ~95% measured recall, ~1,200 candidates/entity
- **`memory_efficient_submission.py`** (the shipped script): Single-pass 3-char prefix blocking, completely different code
- The 95% recall number didn't describe what was actually running

### Fix
**File:** `scripts/memory_efficient_submission_v2.py`

Rewrote the submission script to use the actual 4-pass blocking logic:
1. Name prefix (4 chars) + country
2. Name token overlap + country  
3. Address numeric (first number) + country
4. Address token overlap + country

All passes union together, matching the tested `blocking.py` behavior.

### Verification Needed
```bash
cd code/business_entity_resolution/scripts
python memory_efficient_submission_v2.py
# Compare candidate counts with blocking.py output on same data
```

---

## Issue #2: Top 50 Selection Not Sorted 🔴 CRITICAL → ✅

### Problem
```python
# OLD CODE (memory_efficient_submission.py:44)
filtered = target[...].head(50)  # Takes first 50 rows in arbitrary pandas order!
```

For common name prefixes (e.g., "acm", "sta") with thousands of candidates, the true match could be candidate #300 and never reach the matcher. This works fine on small test sets but fails silently at scale.

### Fix
**File:** `scripts/memory_efficient_submission_v2.py`

Before capping at 50, candidates are now **sorted by a fast similarity score**:
```python
quick_score = 0.6 * name_jaro_winkler + 0.4 * addr_jaro_winkler
scored.sort(key=lambda x: x[1], reverse=True)
ranked_candidates = scored[:50]  # Top 50 by similarity, not by row order
```

### Impact
This is the highest-risk bug. Could cost 10-20 percentage points of recall on common business names.

---

## Issue #3: No Measured F0.5 ❌ → ✅

### Problem
- "0.75–0.85" was an assumption, not a measurement
- `scorer.py` exists and is validated, but was never run on actual pipeline predictions
- Thresholds (name ≥85, address ≥70) were chosen without validation on this dataset

### Fix
**File:** `scripts/validate_pipeline.py`

New validation script that:
1. Splits training data into train/val (default 80/20)
2. Runs the actual pipeline on the validation set
3. Computes **real macro F0.5** using `scorer.py`
4. Provides error analysis (worst entities, false positives/negatives)
5. Tests threshold sensitivity

### Usage
```bash
cd code/business_entity_resolution/scripts

# Basic validation with current thresholds
python validate_pipeline.py

# Test different thresholds
python validate_pipeline.py --name_thresh 80 --addr_thresh 65

# Measure impact of conflict resolution
python validate_pipeline.py --no_conflict_resolution
```

### Next Steps
1. **Run this ASAP** to get the real F0.5 number
2. Adjust thresholds based on results
3. If F0.5 < 0.70, investigate feature engineering or blocking recall

---

## Issue #4: No Conflict Resolution ❌ → ✅

### Problem
Nothing prevents two different S1 entities from claiming the same S2/S3 record. This violates the entity resolution assumption (one true match) and inflates precision artificially.

### Fix
**Files:** 
- `scripts/memory_efficient_submission_v2.py`
- `scripts/validate_pipeline.py`

Added conflict resolution phase:
1. After generating all matches, identify S2/S3 records claimed by multiple S1 entities
2. For each conflict, compute confidence score: `name_similarity + address_similarity`
3. Keep the match only for the **highest-confidence S1 claimant**
4. Drop it from all other claimants

### Impact
May slightly reduce match count but improves precision and prevents double-counting.

---

## Issue #5: Memory Assumptions Not Verified ⚠️ → Needs Testing

### Problem
- S1 is chunked (5,000 entities at a time)
- But S2 (4.9M rows) and S3 (5.1M rows) are held fully in memory for the entire run
- Peak memory usage was assumed safe, not measured

### Fix
**Partial:** Reduced chunk size to 2,000 in v2 due to multi-pass blocking overhead

### Verification Needed
```bash
# On Windows, open Task Manager and watch memory usage
# On Linux/Mac, run with:
htop  # or top

# Then in another terminal:
cd code/business_entity_resolution/scripts
python memory_efficient_submission_v2.py
```

**Watch for:**
- Peak memory usage > available RAM → will OOM kill
- If OOM occurs, reduce chunk size further or implement S2/S3 chunking

---

## Laya Integration (Bonus)

### File
`src/features_with_laya.py`

### Status
**Stub created, NOT wired up**

### What's There
- Integration point for Laya decision model as one feature in XGBoost
- `laya_score()` function with proper error handling (returns 0.5 neutral on failure)
- Feature engineering pipeline that adds `laya_score` column

### To Complete
1. Install Laya: `pip install laya` (or actual package name)
2. Wire up model loading at top of file:
   ```python
   from laya import load
   _laya_model = load("convaiinnovations/laya")
   _LAYA_ENABLED = True
   ```
3. Fill in the actual inference call in `laya_score()` function
4. Run Model A (features only) vs Model C (features + Laya) ablation test
5. **Only keep if F0.5 improves** — Laya's zero-shot accuracy is below baseline, needs empirical validation

### Expected Performance
- Zero-shot calibration: ECE 0.466 (poor, needs temperature scaling)
- Zero-shot accuracy: 0.362 (below majority baseline)
- Latency: ~33ms per call (excellent, non-autoregressive)

Don't assume it helps — test it.

---

## Test Plan

### Phase 1: Validation (Do This First)
```bash
cd code/business_entity_resolution/scripts

# 1. Measure actual F0.5
python validate_pipeline.py

# 2. Check memory usage during full run
python memory_efficient_submission_v2.py
# Monitor Task Manager / htop
```

**Decision point:** If F0.5 < 0.70 or memory OOMs, investigate before proceeding.

### Phase 2: Threshold Optimization
```bash
# Test threshold grid
for name_t in 80 82 85 87 90; do
  for addr_t in 65 70 75; do
    echo "Testing name=$name_t addr=$addr_t"
    python validate_pipeline.py --name_thresh $name_t --addr_thresh $addr_t
  done
done
```

Find the (name_thresh, addr_thresh) pair with highest F0.5.

### Phase 3: Full Test Submission
```bash
# Run fixed pipeline on full test set with optimal thresholds
python memory_efficient_submission_v2.py

# Validate output format
python ../../../student_resource/utils/validate_submission.py \
  --candidate_pairs ../../../output/candidate_pairs.tsv \
  --matching_results ../../../output/matching_results.tsv
```

### Phase 4: Laya Ablation (Optional)
Only if time permits and Phase 1-3 results are solid.

---

## Files Changed/Added

### New Files
- ✅ `scripts/memory_efficient_submission_v2.py` - Fixed submission script
- ✅ `scripts/validate_pipeline.py` - F0.5 measurement on validation set
- ✅ `src/features_with_laya.py` - Laya integration stub

### Unchanged (Still Valid)
- ✅ `src/blocking.py` - Multi-pass blocking (reference implementation)
- ✅ `src/features.py` - Original feature engineering
- ✅ `src/scorer.py` - Official F0.5 metric
- ✅ `src/normalization.py` - Text normalization utilities

### Deprecated
- ❌ `scripts/memory_efficient_submission.py` - OLD, has all 5 bugs

---

## Risk Assessment

| Issue | Severity | Fixed | Verified |
|-------|----------|-------|----------|
| #1: Blocking mismatch | High | ✅ | ⏳ Needs test run |
| #2: Top 50 unsorted | **CRITICAL** | ✅ | ⏳ Needs test run |
| #3: No F0.5 measurement | High | ✅ | ⏳ Run validate_pipeline.py |
| #4: No conflict resolution | Medium | ✅ | ⏳ Needs test run |
| #5: Memory assumptions | Medium | ⚠️ Partial | ⏳ Monitor during run |

---

## Next Actions (Priority Order)

1. **IMMEDIATE:** Run `validate_pipeline.py` to get real F0.5 on validation data
2. **IMMEDIATE:** Check if current thresholds (85/70) are optimal or need tuning
3. **HIGH:** Run `memory_efficient_submission_v2.py` on full test set and monitor memory
4. **HIGH:** Validate output format with official validator
5. **MEDIUM:** If F0.5 < 0.70, investigate blocking recall or feature engineering
6. **LOW:** Laya integration (only if time permits after 1-5 are solid)

---

## Performance Expectations

### Conservative Estimates (After Fixes)
- **Blocking recall:** 92-97% (multi-pass with top-50 ranking)
- **Matcher precision:** 85-92% (with conflict resolution)
- **Matcher recall:** 75-85% (depends on thresholds)
- **Macro F0.5:** 0.70-0.80 (weighted toward precision)

### High-Risk Scenarios
- Common business names ("ABC Inc", "Star Ltd") → may still hit top-50 cap
- Addresses with typos/variations → may miss on address threshold
- Cross-country near-duplicates → blocked by country filter

---

## Questions to Answer with Validation

1. What's the **actual F0.5** on held-out data?
2. What's the precision/recall breakdown?
3. Which S1 entities perform worst (error analysis)?
4. Are thresholds (85/70) optimal for this dataset?
5. How much does conflict resolution help (+X% precision)?
6. What's the peak memory usage at scale?

**Run `validate_pipeline.py` to answer 1-5. Monitor memory during full run for #6.**

---

## Contact / Escalation

If validation F0.5 < 0.65:
- Re-examine blocking strategy (may need 5th pass or lower top-N cap)
- Consider ML-based matcher instead of rule thresholds
- Check for data quality issues (normalization failures)

If memory OOMs:
- Chunk S2/S3 as well (not just S1)
- Use disk-backed DataFrames (Dask or similar)
- Move to higher-memory machine

---

**Status:** All fixes implemented, validation scripts ready. **Run tests now.**
