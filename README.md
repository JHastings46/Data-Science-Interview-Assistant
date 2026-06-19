# Data Science Interview Assistant

An end-to-end LLM project that fine-tunes FLAN-T5 on data science 
interview questions and deploys a RAG-powered answer generator.

## Live Demo
🚀 [Try the app here](https://huggingface.co/spaces/joelhastings-ds/ds-interview-assistant)

## Project Overview

| Stage | What I Did |
|---|---|
| Lab 1 | Prompt engineering — zero/one/few shot inference |
| Lab 2 | Full fine-tuning + PEFT/LoRA on 683 Q&A pairs |
| Lab 3 | PPO-based quality tuning (RLHF-style) |
| Deployment | RAG pipeline deployed as Streamlit app |

## Results

### Fine-Tuning (Lab 2)
| Model | rouge1 | rouge2 | rougeL |
|---|---|---|---|
| Base flan-t5 | 0.256 | 0.086 | 0.208 |
| Full fine-tune | 0.372 | 0.188 | 0.331 |
| PEFT/LoRA | 0.371 | 0.187 | 0.331 |

PEFT matched full fine-tuning at 1% of the parameters.

### RAG vs No RAG
**Question:** What is gradient boosting in machine learning?

**Without RAG:** "a concept known as psychoanalysis where neural 
networks convert upwards one out of two examples"

**With RAG:** "a technique for regression and classification that 
produces a prediction model in the form of an ensemble of weak 
prediction models, typically decision trees"

## Tech Stack
- Model: google/flan-t5-base
- Fine-tuning: PEFT/LoRA (peft==0.9.0)
- Alignment: PPO via TRL
- Retrieval: sentence-transformers/all-MiniLM-L6-v2
- Deployment: Streamlit on Hugging Face Spaces
- Dataset: 683 combined Q&A pairs from HuggingFace

## Model Checkpoint
[joelhastings-ds/ds-interview-assistant](https://huggingface.co/joelhastings-ds/ds-interview-assistant)

## Notebooks
- `1. DataAssistant.ipynb` — Prompt engineering
- `2. DA_Finetuning_.ipynb` — Full fine-tune + PEFT
- `3. RLHF Detoxification.ipynb` — PPO quality tuning
