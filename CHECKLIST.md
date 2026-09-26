# Pre-Submission Checklist

**Date:** 2026-09-26  
**Time:** 05:40 UTC

---

## ✅ What's Been Fixed

- [x] **Issue #1:** Reconciled blocking implementations - submission script now uses 4-pass blocking
- [x] **Issue #2:** Fixed top-50 selection - candidates sorted by similarity BEFORE capping
- [x] **Issue #3:** Created validation script to measure actual F0.5 (not assumed)
- [x] **Issue #4:** Added conflict resolution (highest confidence wins)
- [x] **Issue #5:** Reduced chunk size for memory efficiency (monitoring still needed)
- [x] Created comprehensive documentation (PIPELINE_FIXES.md, FIXES_SUMMARY.md)
- [x] Created quick-start scripts (run_validation.bat, run_validation.sh)
- [x] Created Laya integration stub (optional, not required)

---

## 📋 Critical Path - Do These In Order

### Step 1: Validate the Fixes (15-20 minutes)
```bash
cd code/business_entity_resolution/scripts
python validate_pipeline.py
```

**What to look for:**
- [ ] Script runs without errors
- [ ] Macro F0.5 score is displayed
- [ ] Precision/recall breakdown shown
- [ ] Error analysis shows worst entities

**Decision point:**
- If F0.5 ≥ 0.70 → Proceed to Step 2
- If 0.60 ≤ F0.5 < 0.70 → Try threshold tuning (Step 1a)
- If F0.5 < 0.60 → Investigate blocking/features before proceeding

### Step 1a: Threshold Tuning (Optional, 10-15 minutes)
```bash
# Try different threshold combinations
python validate_pipeline.py --name_thresh 80 --addr_thresh 65
python validate_pipeline.py --name_thresh 82 --addr_thresh 70
python validate_pipeline.py --name_thresh 88 --addr_thresh 75
```

**Goal:** Find (name_thresh, addr_thresh) pair with highest F0.5

---

### Step 2: Full Test Submission (30-45 minutes)

**Before starting:**
- [ ] Note current time and available memory
- [ ] Open Task Manager (Windows) or htop (Linux/Mac) to monitor memory
- [ ] Have optimal thresholds from Step 1

```bash
cd code/business_entity_resolution/scripts
python memory_efficient_submission_v2.py
```

**Monitor during run:**
- [ ] Peak memory usage stays below available RAM
- [ ] No OOM errors
- [ ] Progress messages appear (chunked processing)
- [ ] Conflict resolution summary shows up

**Expected outputs:**
- [ ] `output/candidate_pairs.tsv` created
- [ ] `output/matching_results.tsv` created
- [ ] Both files non-empty

---

### Step 3: Validate Output Format (2 minutes)
```bash
cd student_resource/utils
python validate_submission.py \
  --candidate_pairs ../../output/candidate_pairs.tsv \
  --matching_results ../../output/matching_results.tsv
```

**What to check:**
- [ ] Validation passes with no errors
- [ ] All S1 entities present in output
- [ ] Format matches competition requirements

---

### Step 4: Submit (2 minutes)
- [ ] Upload `candidate_pairs.tsv` to competition platform
- [ ] Upload `matching_results.tsv` to competition platform
- [ ] Note submission timestamp
- [ ] Wait for leaderboard score

---

## 🚨 Troubleshooting

### Validation F0.5 < 0.60
**Possible causes:**
1. Blocking recall too low (true matches not in candidate set)
2. Thresholds too strict
3. Data quality issues (normalization failing)

**Actions:**
```bash
# Check blocking recall by inspecting candidates
# Review error analysis output - which entities fail worst?
# Try much lower thresholds temporarily to isolate issue
python validate_pipeline.py --name_thresh 70 --addr_thresh 60
```

---

### Memory OOM During Full Run
**Symptoms:** Python crashes with "MemoryError" or OS kills the process

**Actions:**
1. Reduce S1 chunk size in `memory_efficient_submission_v2.py`:
   ```python
   s1_chunk_size = 1000  # Instead of 2000
   ```
2. Consider chunking S2/S3 as well (more complex)
3. Close other applications to free RAM
4. Run on machine with more memory if available

---

### Output Validation Fails
**Common issues:**
- Missing S1 entities in output → check loop completion
- Wrong delimiter (comma vs tab) → must be tab-separated
- Empty candidate/match sets → check threshold logic

**Fix:** Review the validation error message and corresponding line in output files

---

## 📊 Expected Performance (Post-Fix)

### Conservative Estimates
| Metric | Before Fixes | After Fixes | Improvement |
|--------|--------------|-------------|-------------|
| Blocking recall | ~85% (1-pass) | ~95% (4-pass) | +10% |
| Matcher recall | ~65% (unsorted) | ~78% (sorted) | +13% |
| Matcher precision | ~88% | ~90% (conflicts) | +2% |
| **Macro F0.5** | **~0.55** | **~0.72** | **+17 pts** |

### Reality Check
- Validation will give you the **actual** numbers
- If below estimates, thresholds may need tuning
- If significantly above, great! Don't second-guess it.

---

## ⏱️ Time Budget

| Task | Estimated Time | Status |
|------|----------------|--------|
| Validation run | 15-20 min | ⏳ Not started |
| Threshold tuning (optional) | 10-15 min | ⏳ Not started |
| Full test submission | 30-45 min | ⏳ Not started |
| Output validation | 2 min | ⏳ Not started |
| Upload submission | 2 min | ⏳ Not started |
| **Total** | **~1 hour** | ⏳ |

**Laya integration (optional):** +2-3 hours if pursuing

---

## 🎯 Success Criteria

Before submission:
- [x] All 5 critical issues documented and fixed
- [ ] Validation F0.5 measured and ≥ 0.70
- [ ] Full test run completes without OOM
- [ ] Output passes format validation
- [ ] Ready to submit

After submission:
- [ ] Leaderboard score ≥ 0.65 (public test set)
- [ ] No format errors from platform
- [ ] Score within expected range of validation F0.5

---

## 📝 Notes Section

**Validation F0.5 (from Step 1):** _____________

**Optimal thresholds:** name = _____, address = _____

**Peak memory usage (from Step 2):** _____________

**Full run time:** _____ minutes

**Leaderboard score (after submission):** _____________

---

## 🚀 Ready to Start?

**Your next action:**
```bash
# Run this command now
cd code/business_entity_resolution/scripts
python validate_pipeline.py
```

This will measure your actual F0.5 and tell you if the fixes worked.

**Good luck! The pipeline is ready.**

---

## 📚 Documentation Reference

- **FIXES_SUMMARY.md** - Executive summary of all fixes
- **PIPELINE_FIXES.md** - Detailed technical documentation
- **run_validation.bat/.sh** - Quick-start script
- This checklist - Step-by-step execution guide

**Current Status:** All fixes implemented ✅ | Validation pending ⏳
