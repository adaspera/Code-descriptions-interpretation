# Merges LoRA adapters with base model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"
FINETUNED_MODEL_PATH = "./qwen1.7B-feedback-finetuned-all/checkpoint-1000"
MERGED_MODEL_PATH = "./Qwen3-merged-10000"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)