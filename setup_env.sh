#!/bin/bash

# Create conda environment
conda env create -f environment.yml

# Activate environment and install NLP models
eval "$(conda shell.bash hook)"
conda activate GUIDE

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Setup complete. Activate with: conda activate GUIDE" 