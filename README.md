# 🦙 Llama-3.2-3B QLoRA Fine-Tuning on Dolly-15K

Production-grade QLoRA fine-tuning pipeline for Llama-3.2-3B-Instruct  
using Unsloth, HuggingFace TRL, and MLflow experiment tracking.

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Mervecaliskan%2Fllama3.2--3b--dolly--qlora-blue)](https://huggingface.co/Mervecaliskan/llama3.2-3b-dolly-qlora)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%2B%20TRL-orange)](https://github.com/unslothai/unsloth)

---

## What This Project Does

Fine-tunes **Llama-3.2-3B-Instruct** on a 500-example subset of  
[Databricks Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)  
using QLoRA — only 1.30% of parameters are trained.

**Key results:**
- Train loss: 1.87 → 1.49
- Val loss: 1.77 → 1.57 — no overfitting
- Training time: ~6 minutes on T4 GPU
- GPU peak memory: 7.95 GB

---

## Project Structure

```
llama3-qlora-finetuning/
├── QLoRA_Dolly_FineTuning_v3.ipynb   # Main training notebook
├── README.md
└── requirements.txt
```

---

## Notebook Overview

| Section | Content |
|---------|---------|
| 1. Setup | Install unsloth, trl, peft, mlflow |
| 2. GPU Check | T4 verification, HuggingFace login |
| 3. Model Loading | Llama-3.2-3B-Instruct, 4-bit NF4 quantization |
| 4. LoRA Adapters | r=16, 7 target modules, 24.3M trainable params |
| 5. Dataset | Dolly-15K, 500 examples, Llama-3 instruct format |
| 6. MLflow Setup | Experiment tracking, hyperparameter logging, env tags |
| 7. Training | SFTTrainer, 120 steps, step-level loss logging |
| 8. Model Testing | open_qa, classification, summarization, brainstorming |
| 9. HuggingFace Push | LoRA adapter upload to Hub |

---

## Training Configuration

```python
# QLoRA config
lora_r         = 16
lora_alpha     = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
quantization   = "4-bit NF4"

# Training config
learning_rate  = 2e-4
lr_scheduler   = "cosine"
optimizer      = "adamw_8bit"
max_steps      = 120
effective_batch = 8  # 2 × 4 grad accumulation
```

---

## Training Results

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 30   | 1.8728    | 1.7720   |
| 60   | 1.7791    | 1.5834   |
| 90   | 1.7449    | 1.5770   |
| 120  | **1.4911** | **1.5746** ✅ |

Val loss decreases consistently — no overfitting on 500 examples.

---

## MLflow Tracking

```
Experiment : llama3-dolly-finetuning
Run        : qlora-r16-lr2e4-dolly500

Logged params  : lora_r, lora_alpha, lr, scheduler, optimizer,
                 batch_size, max_steps, dataset, train_examples
Logged metrics : train_loss (per step), val_loss (per step),
                 final_train_loss, best_train_loss, best_val_loss,
                 training_minutes, gpu_peak_memory_gb
Logged tags    : developer, environment, gpu, framework, task
```

---

## Quick Start

```bash
# Open in Google Colab (T4 GPU recommended)
# File → Open notebook → GitHub → paste repo URL

# Or run locally
pip install unsloth trl peft transformers bitsandbytes accelerate mlflow
jupyter notebook QLoRA_Dolly_FineTuning_v3.ipynb
```

---

## Use the Fine-Tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", load_in_4bit=True, device_map="auto"
)
model     = PeftModel.from_pretrained(model, "Mervecaliskan/llama3.2-3b-dolly-qlora")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

🤗 Model: [huggingface.co/Mervecaliskan/llama3.2-3b-dolly-qlora](https://huggingface.co/Mervecaliskan/llama3.2-3b-dolly-qlora)

---

## Tech Stack

`PyTorch` · `Unsloth` · `HuggingFace TRL` · `PEFT` · `MLflow` · `bitsandbytes` · `Google Colab T4`

---

*Portfolio project — AI Engineering · Merve Caliskan · 2026*
