@echo off
REM Quick validation script for Windows
REM Usage: run_validation.bat

echo ==========================================
echo PIPELINE VALIDATION - Quick Start
echo ==========================================
echo.

cd /d "%~dp0\code\business_entity_resolution\scripts"

REM Check if training data exists
set TRAIN_PATH=..\..\..\student_resource\dataset\train
if not exist "%TRAIN_PATH%\train_source1.tsv" (
    echo ERROR: Training data not found at %TRAIN_PATH%
    echo Please ensure the dataset is in the correct location.
    exit /b 1
)

echo Step 1: Running validation to measure actual F0.5...
echo ----------------------------------------
python validate_pipeline.py --split_ratio 0.8

echo.
echo ==========================================
echo Validation complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Review the F0.5 score above
echo   2. If F0.5 ^< 0.70, try adjusting thresholds:
echo      python validate_pipeline.py --name_thresh 80 --addr_thresh 65
echo   3. Once satisfied, run full test submission:
echo      python memory_efficient_submission_v2.py
echo.
echo See PIPELINE_FIXES.md for detailed documentation.
