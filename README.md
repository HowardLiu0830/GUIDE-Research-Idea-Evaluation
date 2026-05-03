# GUIDE - Towards Scalable Advising for Research Ideas

![Conference](https://img.shields.io/badge/ACL-2026-blue) [![Paper](https://img.shields.io/badge/arXiv-2507.08870-b31b1b.svg)](https://arxiv.org/abs/2507.08870) [![Demo](https://img.shields.io/badge/Demo-researchguide.work-brightgreen)](https://www.researchguide.work/)

**Accepted at ACL 2026 Main Conference.**

## 🚀 Quick Start (4 Steps)

### 1. Setup Environment
```bash
bash setup_env.sh
conda activate GUIDE
```

### 2. Download Research Databases
Run this first to download the required research paper databases from Hugging Face:
```bash
python database/db_loader.py --action download
```

This downloads 4 databases (~500MB total):
- **abstract_db** - Paper abstracts for similarity search
- **contribution_db** - Paper contributions for similarity search
- **method_db** - Paper methods for similarity search
- **experiment_db** - Paper experiments for similarity search

**Note**: You need an OpenAI API key to use these databases. The download step just fetches the data - the API key is used later for embeddings.

### 3. Configure API Keys
Edit the OpenAI API key at the top of `run_pipeline.sh`:
```bash
OPENAI_KEY="your-actual-openai-key-here"
```

The pipeline now relies on OpenAI only (used for both embeddings and the advising LLM), so no other provider keys are needed.

### 4. Run the System
```bash
bash run_pipeline.sh
```
A timestamped log will be written to `logs/run_pipeline_<timestamp>.log`.



## 🔧 Configuration Options

### Available AI Models
Only OpenAI GPT models are supported:
- `gpt-5.4-mini` (default)
- `gpt-5.4-nano`
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4.1-nano`

See `python advising_gen.py --help` for the canonical list.

### Processing Range
Edit `run_pipeline.sh` to change which papers to process:
```bash
START_IDX=0    # Start from paper 0
END_IDX=2      # Process up to paper 2 (demo setting)
```

### Paper Sections to Analyze
```bash
PAPER_SECTIONS="introduction,method,experiments"
SEARCH_TYPES="abstract,contribution,method,experiments"
```

### Related-Paper Retrieval
`prompt_gen.py` retrieves related work from the four Chroma databases, then applies two filters:
- **Year filter** (`--year_threshold`, default `2023`) — only keeps papers with `year <= threshold` in the DB metadata.
- **ROUGE-L dedup** (`--rouge_threshold`, default `0.5`) — drops retrieved papers whose abstract is too similar to the target abstract (helps avoid the target paper itself / near-duplicates being returned as "prior work").

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
- **`advising.json`** - Full structured advising output with metadata
- **`advising_clean.json`** - Clean format for easy reading
- **`token_usage.json`** - API usage statistics (if `--track_tokens` enabled)

### Advising Schema
Each entry in `advising.json` is a JSON object with the following fields:
- `summary` *(string)* — concise paragraph summarizing the paper
- `comparison_with_previous_work` *(list of exactly 5 strings)* — each item references a prior work by its **paper title** and contains exactly two sentences (what the prior work does + how it relates to the target paper). No URLs.
- `Novelty`, `Significance`, `Soundness`, `strengths`, `weaknesses`, `Suggestion` *(each a list of exactly 4 strings)* — balanced, multi-sentence assessments

The legacy single-string `Evaluation` field has been removed.

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
  --end_idx 10 \
  --year_threshold 2023 \
  --rouge_threshold 0.5 \
  --paper_sections introduction,method,experiments \
  --search_types abstract,contribution,method,experiments
```

### Generate Advising Only
```bash
python advising_gen.py \
  --openai_key your_key \
  --paper_path data/sample.json \
  --output_path advising.json \
  --output_path_clean advising_clean.json \
  --prompt_path prompts.jsonl \
  --model gpt-5.4-mini \
  --start_idx 0 \
  --end_idx 10 \
  --track_tokens
```

## BibTex

Please cite our work if you find the package useful 😄
```
@article{liu2025guide,
  title={GUIDE: Towards Scalable Advising for Research Ideas},
  author={Liu, Yaowenqi and Meng, Bingxu and Pan, Rui and Liu, Yuxing and Huang, Jerry and You, Jiaxuan and Zhang, Tong},
  journal={arXiv preprint arXiv:2507.08870},
  year={2025}
}
```
