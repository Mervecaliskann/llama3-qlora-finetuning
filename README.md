# 🦙 Llama-3.2-3B QLoRA Fine-Tuning × Dolly-15K

Fine-tuned Llama-3.2-3B-Instruct using QLoRA on a 500-example subset
of the Databricks Dolly-15K dataset.

## Results

| Step | Train Loss | Val Loss |
|---|---|---|
| 30  | 1.872 | 1.772 |
| 60  | 1.779 | 1.583 |
| 90  | 1.744 | 1.577 |
| 120 | 1.491 | 1.574 |

Train loss: 1.87 → 1.49 | Val loss: 1.77 → 1.57 | No overfitting ✅

## Method

- **Model:** `unsloth/Llama-3.2-3B-Instruct`
- **Quantization:** 4-bit NF4 (QLoRA)
- **LoRA:** r=16, alpha=16, target: q/k/v/o/gate/up/down_proj
- **Dataset:** Databricks Dolly-15K — 500 examples, 8 categories
- **Tracking:** MLflow
- **Hardware:** Google Colab T4 GPU

## Stack

PyTorch · Unsloth · HuggingFace TRL · SFTTrainer · MLflow

## Files

| File | Description |
|---|---|
| `QLoRA_Dolly_FineTuning.ipynb` | Full training notebook |

## 🤗 Model on HuggingFace

[Mervecaliskan/llama3.2-3b-dolly-qlora](https://huggingface.co/Mervecaliskan/llama3.2-3b-dolly-qlora)

## Quick Start
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base  = "meta-llama/Llama-3.2-3B-Instruct"
adapter = "Mervecaliskan/llama3.2-3b-dolly-qlora"

tokenizer = AutoTokenizer.from_pretrained(base)
model     = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True)
model     = PeftModel.from_pretrained(model, adapter)
```
