#!/bin/bash
#
# GUIDE Research Pipeline Runner
# Combines prompt_gen.py and advising_gen.py for paper analysis.
#
set -euo pipefail

# === Configuration ===
OPENAI_KEY="your-actual-openai-key-here"

PAPER_PATH="data/sample.json"
PROMPT_PATH="prompts.jsonl"
ADVISING_PATH="advising.json"
ADVISING_CLEAN_PATH="advising_clean.json"
TOKEN_OUTPUT="token_usage.json"

# Prompt generation
NUM_RELATED=3
START_IDX=0
END_IDX=2  # Process first 2 papers for demo
YEAR_THRESHOLD=2023
ROUGE_THRESHOLD=0.5
PAPER_SECTIONS="introduction,method,experiments"
SEARCH_TYPES="abstract,contribution,method,experiments"

# Advising generation
MODEL="gpt-5.4-mini"

# === Logging ===
mkdir -p logs
LOG_FILE="logs/run_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

# === Sanity checks ===
if [ ! -f "$PAPER_PATH" ]; then
    echo "❌ $PAPER_PATH not found"
    exit 1
fi

echo "🚀 GUIDE Research Pipeline"
echo "=================================================="
echo "📊 Processing sample papers from $PAPER_PATH"
echo "🔧 Edit OPENAI_KEY at the top of this script before running"
echo

# === Step 1: Generate prompts ===
echo "🔍 Generating prompts..."
python prompt_gen.py \
    --openai_key "$OPENAI_KEY" \
    --paper_path "$PAPER_PATH" \
    --output_path "$PROMPT_PATH" \
    --num_related "$NUM_RELATED" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --year_threshold "$YEAR_THRESHOLD" \
    --rouge_threshold "$ROUGE_THRESHOLD" \
    --paper_sections "$PAPER_SECTIONS" \
    --search_types "$SEARCH_TYPES"
echo "✅ Prompts generated -> $PROMPT_PATH"
echo

# === Step 2: Generate advising ===
echo "📝 Generating advising..."
python advising_gen.py \
    --openai_key "$OPENAI_KEY" \
    --paper_path "$PAPER_PATH" \
    --output_path "$ADVISING_PATH" \
    --output_path_clean "$ADVISING_CLEAN_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --model "$MODEL" \
    --track_tokens \
    --token_output "$TOKEN_OUTPUT"
echo "✅ Advising generated -> $ADVISING_PATH, $ADVISING_CLEAN_PATH"
echo "✅ Pipeline complete!"
