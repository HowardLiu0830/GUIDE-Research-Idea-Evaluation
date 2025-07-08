#!/usr/bin/env python3
"""
GUIDE Research Pipeline Runner
Combines prompt_gen.py and review_gen.py for paper analysis
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if conda environment is activated"""
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        print("⚠️  Please activate conda environment: conda activate GUIDE")
        return False
    
    if os.environ['CONDA_DEFAULT_ENV'] != 'GUIDE':
        print(f"⚠️  Wrong environment '{os.environ['CONDA_DEFAULT_ENV']}'. Please run: conda activate GUIDE")
        return False
    
    return True

def run_prompt_generation():
    """Generate prompts for papers"""
    print("🔍 Generating prompts...")
    
    cmd = [
        'python', 'prompt_gen.py',
        '--openai_key', 'your openai key',
        '--paper_path', 'data/sample.json',
        '--output_path', 'prompts.jsonl',
        '--num_related', '3',
        '--start_idx', '0',
        '--end_idx', '2',  # Process first 2 papers for demo
        '--paper_sections', 'introduction,method,experiments',
        '--search_types', 'abstract,contribution,method,experiments'
    ]
    
    return subprocess.run(cmd, capture_output=True, text=True)

def run_review_generation():
    """Generate reviews from prompts"""
    print("📝 Generating reviews...")
    
    cmd = [
        'python', 'review_gen.py',
        '--openai_key', 'your openai key',
        '--google_key', 'your google key',  # Replace with actual key
        '--deepinfra_key', 'your_deepinfra_key',  # Replace with actual key
        '--paper_path', 'data/sample.json',
        '--output_path', 'reviews.json',
        '--output_path_clean', 'reviews_clean.json',
        '--prompt_path', 'prompts.jsonl',
        '--start_idx', '0',
        '--end_idx', '2',  # Process first 2 papers for demo
        '--model', 'gpt-4o-mini',
        '--track_tokens'
    ]
    
    return subprocess.run(cmd, capture_output=True, text=True)

def main():
    """Main pipeline execution"""
    print("🚀 GUIDE Research Pipeline")
    print("=" * 50)
    
    # Check environment
    if not check_requirements():
        sys.exit(1)
    
    # Check data file exists
    if not Path('data/sample.json').exists():
        print("❌ data/sample.json not found")
        sys.exit(1)
    
    print("📊 Processing sample papers from data/sample.json")
    print("🔧 Modify API keys in this script before running")
    print()
    
    # Step 1: Generate prompts
    result1 = run_prompt_generation()
    if result1.returncode != 0:
        print("❌ Prompt generation failed:")
        print(result1.stderr)
        sys.exit(1)
    
    print("✅ Prompts generated -> prompts.jsonl")
    
    # Step 2: Generate reviews
    result2 = run_review_generation()
    if result2.returncode != 0:
        print("❌ Review generation failed:")
        print(result2.stderr)
        sys.exit(1)
    
    print("✅ Reviews generated -> reviews.json, reviews_clean.json")
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    main() 