#!/bin/bash
# Quick validation script - Run this to test all fixes
# Usage: bash run_validation.sh

set -e  # Exit on error

echo "=========================================="
echo "PIPELINE VALIDATION - Quick Start"
echo "=========================================="
echo ""

cd "$(dirname "$0")/code/business_entity_resolution/scripts"

# Check if training data exists
TRAIN_PATH="../../../student_resource/dataset/train"
if [ ! -f "$TRAIN_PATH/train_source1.tsv" ]; then
    echo "ERROR: Training data not found at $TRAIN_PATH"
    echo "Please ensure the dataset is in the correct location."
    exit 1
fi

echo "Step 1: Running validation to measure actual F0.5..."
echo "----------------------------------------"
python validate_pipeline.py --split_ratio 0.8

echo ""
echo "=========================================="
echo "Validation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the F0.5 score above"
echo "  2. If F0.5 < 0.70, try adjusting thresholds:"
echo "     python validate_pipeline.py --name_thresh 80 --addr_thresh 65"
echo "  3. Once satisfied, run full test submission:"
echo "     python memory_efficient_submission_v2.py"
echo ""
echo "See PIPELINE_FIXES.md for detailed documentation."
