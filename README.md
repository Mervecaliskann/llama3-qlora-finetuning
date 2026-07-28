# Llama-3.2-3B QLoRA Fine-Tuning on Dolly-15K
 
A QLoRA fine-tuning pipeline for Llama-3.2-3B-Instruct, using Unsloth, HuggingFace TRL, and MLflow for experiment tracking. The whole thing runs on a single free T4 GPU in about six minutes.
 
## What it does
 
Fine-tunes Llama-3.2-3B-Instruct on a 500-example subset of [Databricks Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) using QLoRA — the base model is frozen and quantized to 4-bit, and only small LoRA adapters are trained. That comes out to about **0.75% of the parameters** (24.3M out of 3.24B).
 
**Results (120 steps):**
 
| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 30 | 1.8729 | 1.7720 |
| 60 | 1.7791 | 1.5833 |
| 90 | 1.7449 | 1.5769 |
| 120 | 1.4911 | 1.5746 |
 
Both losses drop and validation loss stays flat at the end, so no overfitting on 500 examples. Training took ~6 minutes on a T4 with ~8 GB peak GPU memory.
 
## How it works
 
The idea behind QLoRA is efficiency. A 3B model normally won't fit comfortably on a small GPU, so:
 
- **4-bit NF4 quantization** shrinks the frozen base model so it fits on a T4.
- **LoRA adapters** (rank `r=16`) are added on top of the frozen weights, and only those are trained — that's the 0.75%.
- Adapters go on 7 modules: the attention projections (`q/k/v/o_proj`) and the feed-forward layers (`gate/up/down_proj`).
## Notebook sections
 
| # | Section |
|---|---------|
| 1 | Setup — install unsloth, trl, peft, mlflow |
| 2 | GPU check + HuggingFace login |
| 3 | Model loading — Llama-3.2-3B, 4-bit NF4 |
| 4 | LoRA adapters — r=16, 7 target modules, 24.3M trainable |
| 5 | Dataset — Dolly, 500 examples (450 train / 50 val), Llama-3 format |
| 6 | MLflow — experiment + hyperparameter/metric logging |
| 7 | Training — SFTTrainer, 120 steps |
| 8 | Model testing — open QA, classification, summarization |
| 9 | Push LoRA adapter to HuggingFace Hub |
 
## Training config
 
```python
lora_r         = 16
lora_alpha     = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
quantization   = "4-bit NF4"
 
learning_rate  = 2e-4
lr_scheduler   = "cosine"
optimizer      = "adamw_8bit"
max_steps      = 120
effective_batch = 8          # 2 × 4 gradient accumulation
```
 
## MLflow tracking
 
Logged params: `lora_r, lora_alpha, lr, scheduler, optimizer, batch_size, max_steps, dataset, train_examples`
Logged metrics: `train_loss, val_loss (per step), final/best losses, training_minutes, gpu_peak_memory_gb`
 
## Using the fine-tuned model
 
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
 
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, "Mervecaliskan/llama3.2-3b-dolly-qlora")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```
 
🤗 Model: [huggingface.co/Mervecaliskan/llama3.2-3b-dolly-qlora](https://huggingface.co/Mervecaliskan/llama3.2-3b-dolly-qlora)
 
## Stack
 
PyTorch · Unsloth · HuggingFace TRL · PEFT · MLflow · bitsandbytes · Google Colab (T4)
 
