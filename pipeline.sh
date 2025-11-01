#!/bin/bash

# Automated Pipeline to Reproduce Figures 2 and 3
# Based on "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions"
# This script generates data, trains models, and evaluates to reproduce paper figures

set -e  # Exit on any error

echo "=================================================="
echo "Instruction Hierarchy: Automated Pipeline"
echo "Reproducing Figures 2 and 3 from the Paper"
echo "=================================================="
echo ""

# ====================
# Configuration
# ====================

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
GENERATION_MODEL="Qwen/Qwen2.5-7B-Instruct"
BASELINE_DIR="./baseline_model"
HIERARCHY_DIR="./instruction_hierarchy_model"
DATA_DIR="./training_data"
EVAL_DIR="./eval_data"
RESULTS_FILE="./evaluation_results.json"
FIGURES_DIR="./figures"

# Data generation settings
NUM_SAMPLES=5000  # Samples per data type
BATCH_SIZE_GEN=4
SEED=42

# ====================
# Step 1: Generate All Training Data
# ====================

echo "=================================================="
echo "STEP 1: Generating Training Data"
echo "=================================================="
echo ""
echo "Generating BOTH baseline and instruction hierarchy data..."
echo "  - Baseline data: ${NUM_SAMPLES} samples"
echo "  - Instruction hierarchy data: ${NUM_SAMPLES} samples"
echo ""

if [ -f "$DATA_DIR/baseline_data.json" ] && [ -f "$DATA_DIR/instruction_hierarchy_full.json" ]; then
    echo "Training data already exists. Skipping..."
    echo "  ✓ $DATA_DIR/baseline_data.json"
    echo "  ✓ $DATA_DIR/instruction_hierarchy_full.json"
else
    echo "Running generate_data.py with --data_type both..."
    echo "This will take 10-30 minutes depending on hardware..."
    echo ""

    python generate_data.py \
        --data_type both \
        --num_samples $NUM_SAMPLES \
        --output_dir "$DATA_DIR" \
        --generation_model "$GENERATION_MODEL" \
        --batch_size $BATCH_SIZE_GEN \
        --seed $SEED

    # Verify files were created
    if [ ! -f "$DATA_DIR/baseline_data.json" ] || [ ! -f "$DATA_DIR/instruction_hierarchy_full.json" ]; then
        echo "✗ Error: Data generation did not produce expected files"
        exit 1
    fi
fi

echo ""
echo "✓ Training data ready:"
echo "  - Baseline: $DATA_DIR/baseline_data.json"
echo "  - Hierarchy: $DATA_DIR/instruction_hierarchy_full.json"
echo ""

# ====================
# Step 2: Create Evaluation Datasets
# ====================

echo "=================================================="
echo "STEP 2: Creating Evaluation Datasets"
echo "=================================================="
echo ""

if [ -d "$EVAL_DIR" ] && [ "$(ls -A $EVAL_DIR/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Evaluation data already exists. Skipping..."
    echo "Found $(ls -1 $EVAL_DIR/*.json 2>/dev/null | wc -l) evaluation datasets"
else
    echo "Running create_eval_datasets.py..."
    python create_eval_datasets.py \
        --data_dir "$DATA_DIR" \
        --eval_dir "$EVAL_DIR" \
        --eval_split 0.2
fi

echo ""
echo "✓ Evaluation datasets ready in $EVAL_DIR/"
echo ""

# ====================
# Step 3: Train Baseline Model
# ====================

echo "=================================================="
echo "STEP 3: Training Baseline Model"
echo "=================================================="
echo ""
echo "Training on CAPABILITY DATA ONLY (no hierarchy data)"
echo "This is the red bar in Figures 2 & 3"
echo ""

if [ -d "$BASELINE_DIR/final" ]; then
    echo "Baseline model already exists."
    echo "To retrain, delete $BASELINE_DIR and run again."
    echo ""
else
    echo "Running train_baseline.py..."
    echo "Training mode: both (SFT + DPO)"
    echo "This will take 2-4 hours depending on hardware..."
    echo ""

    python train_baseline.py \
        --mode both \
        --data_path "$DATA_DIR/baseline_data.json" \
        --output_dir "$BASELINE_DIR" \
        --model_name "$BASE_MODEL" \
        --num_epochs_sft 3 \
        --num_epochs_dpo 1 \
        --batch_size 1 \
        --gradient_accumulation_steps 8

    # Verify model was created
    if [ ! -d "$BASELINE_DIR/final" ]; then
        echo "✗ Error: Baseline model training did not produce expected output"
        exit 1
    fi
fi

echo ""
echo "✓ Baseline model ready at $BASELINE_DIR/final/"
echo ""

# ====================
# Step 4: Train Instruction Hierarchy Model
# ====================

echo "=================================================="
echo "STEP 4: Training Instruction Hierarchy Model"
echo "=================================================="
echo ""
echo "Training on INSTRUCTION HIERARCHY DATA"
echo "This is the blue bar in Figures 2 & 3"
echo ""

if [ -d "$HIERARCHY_DIR/dpo_final" ]; then
    echo "Instruction hierarchy model already exists."
    echo "To retrain, delete $HIERARCHY_DIR and run again."
    echo ""
else
    echo "Running instruction_hierarchy_training.py..."
    echo "Training mode: both (SFT + DPO)"
    echo "This will take 3-6 hours depending on hardware..."
    echo ""

    python instruction_hierarchy_training.py \
        --mode both \
        --data_path "$DATA_DIR/instruction_hierarchy_full.json" \
        --output_dir "$HIERARCHY_DIR" \
        --model_name "$BASE_MODEL" \
        --num_epochs_sft 5 \
        --num_epochs_dpo 3 \
        --batch_size 1 \
        --gradient_accumulation_steps 8

    # Verify model was created
    if [ ! -d "$HIERARCHY_DIR/dpo_final" ]; then
        echo "✗ Error: Hierarchy model training did not produce expected output"
        exit 1
    fi
fi

echo ""
echo "✓ Instruction hierarchy model ready at $HIERARCHY_DIR/dpo_final/"
echo ""

# ====================
# Step 5: Evaluate Both Models
# ====================

echo "=================================================="
echo "STEP 5: Evaluating Both Models"
echo "=================================================="
echo ""
echo "Running evaluation on all test datasets..."
echo "This will take 1-3 hours depending on hardware..."
echo ""

if [ -f "$RESULTS_FILE" ]; then
    echo "Evaluation results already exist: $RESULTS_FILE"
    read -p "Re-evaluate? This will overwrite existing results. (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping evaluation..."
    else
        rm "$RESULTS_FILE"
        echo "Running evaluate_models.py..."
        python evaluate_models.py \
            --baseline_model "$BASELINE_DIR/final" \
            --hierarchy_model "$HIERARCHY_DIR/dpo_final" \
            --base_model_name "$BASE_MODEL" \
            --eval_dir "$EVAL_DIR" \
            --output_file "$RESULTS_FILE"
    fi
else
    echo "Running evaluate_models.py..."
    python evaluate_models.py \
        --baseline_model "$BASELINE_DIR/final" \
        --hierarchy_model "$HIERARCHY_DIR/dpo_final" \
        --base_model_name "$BASE_MODEL" \
        --eval_dir "$EVAL_DIR" \
        --output_file "$RESULTS_FILE"
fi

echo ""
echo "✓ Evaluation complete: $RESULTS_FILE"
echo ""

# ====================
# Step 6: Generate Figures
# ====================

echo "=================================================="
echo "STEP 6: Generating Figures 2 and 3"
echo "=================================================="
echo ""

echo "Running plot_results.py..."
python plot_results.py \
    --results_file "$RESULTS_FILE" \
    --output_dir "$FIGURES_DIR" \
    --format png

echo ""
echo "✓ Figures generated in $FIGURES_DIR/"
echo ""

# ====================
# Final Summary
# ====================

echo "=================================================="
echo "✓ PIPELINE COMPLETE!"
echo "=================================================="
echo ""
echo "Generated Artifacts:"
echo ""
echo "📁 Data:"
echo "  - Baseline training: $DATA_DIR/baseline_data.json"
echo "  - Hierarchy training: $DATA_DIR/instruction_hierarchy_full.json"
echo "  - Evaluation sets: $EVAL_DIR/"
echo ""
echo "🤖 Models:"
echo "  - Baseline (red bars): $BASELINE_DIR/final/"
echo "  - Hierarchy (blue bars): $HIERARCHY_DIR/dpo_final/"
echo ""
echo "📊 Results:"
echo "  - Evaluation scores: $RESULTS_FILE"
echo "  - Figure 2 (Main): $FIGURES_DIR/figure_2.png"
echo "  - Figure 3 (Generalization): $FIGURES_DIR/figure_3.png"
echo "  - Combined: $FIGURES_DIR/combined_figures.png"
echo ""
echo "📖 Next Steps:"
echo "  1. View figures: ls -lh $FIGURES_DIR/"
echo "  2. Check results: cat $RESULTS_FILE | python -m json.tool"
echo "  3. Compare models: diff <(ls $BASELINE_DIR/final/) <(ls $HIERARCHY_DIR/dpo_final/)"
echo ""
echo "🔄 To rerun:"
echo "  - Regenerate data: rm -rf $DATA_DIR && ./run_pipeline.sh"
echo "  - Retrain models: rm -rf $BASELINE_DIR $HIERARCHY_DIR && ./run_pipeline.sh"
echo "  - Reevaluate: rm $RESULTS_FILE && ./run_pipeline.sh"
echo ""
echo "=================================================="