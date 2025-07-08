# GUIDE - owards Scalable Advising for Research Ideas

## 🚀 Quick Start (3 Steps)

### 1. Setup Environment
```bash
bash setup_env.sh
conda activate GUIDE
```

### 2. Configure API Keys
Edit the API keys in `run_pipeline.py`:
```python
# Line 48: Add your OpenAI API key
'--openai_key', 'sk-proj-your-actual-openai-key-here',

# Line 49: Add your Google API key  
'--google_key', 'your-actual-google-key-here',

# Line 50: Add your DeepInfra API key
'--deepinfra_key', 'your-actual-deepinfra-key-here',
```

### 3. Run the System
```bash
python run_pipeline.py
```



## 🔧 Configuration Options

### Available AI Models
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Google**: `gemini-2.0-flash-exp`, `gemini-2.0-flash-thinking-exp`  
- **DeepInfra**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`
- **More**: See `review_gen.py --help` for full list

### Processing Range
Edit `run_pipeline.py` to change which papers to process:
```python
'--start_idx', '0',    # Start from paper 0
'--end_idx', '2',      # Process up to paper 2 (demo setting)
```

### Paper Sections to Analyze
```python
'--paper_sections', 'introduction,method,experiments',
'--search_types', 'abstract,contribution,method,experiments'
```

## 📁 Input Data Format

Your `data/sample.json` should contain papers like:
```json
[
  {
    "title": "Paper Title Here",
    "abstract": "Paper abstract...",
    "contribution": "Main contributions...",
    "summary": {
      "introduction": "Introduction summary...",
      "method": "Method summary...", 
      "experiments": "Experiment summary..."
    }
  }
]
```

## 📈 Output

The system generates:
- **`prompts.jsonl`** - Generated prompts for each paper
- **`reviews.json`** - Full structured reviews with metadata  
- **`reviews_clean.json`** - Clean format for easy reading
- **`token_usage.json`** - API usage statistics (if `--track_tokens` enabled)

---

## 🛠️ Advanced Usage (Manual Control)

If you need fine-grained control, run components separately:

### Generate Prompts Only
```bash
python prompt_gen.py \
  --openai_key your_key \
  --paper_path data/sample.json \
  --output_path prompts.jsonl \
  --num_related 3 \
  --start_idx 0 \
  --end_idx 10
```

### Generate Reviews Only  
```bash
python review_gen.py \
  --openai_key your_key \
  --google_key your_key \
  --paper_path data/sample.json \
  --output_path reviews.json \
  --output_path_clean reviews_clean.json \
  --prompt_path prompts.jsonl \
  --model gemini-2.0-flash-exp \
  --start_idx 0 \
  --end_idx 10
```

