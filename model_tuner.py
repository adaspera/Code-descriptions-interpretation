import transformers
import torch
from transformers import (DataCollatorForLanguageModeling, Trainer, TrainingArguments, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_NAME = "m-a-p/Code-Feedback"
OUTPUT_DIR = "./qwen1.7B-feedback-finetuned"

BATCH_SIZE = 6
GRAD_ACCUM_STEPS = 2
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs = NUM_EPOCHS,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    report_to="tensorboard",
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(DATASET_NAME)

def format_conversation(example):
    messages = []
    for message in example["messages"]:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

train_dataset = dataset["train"]
train_dataset = train_dataset.map(format_conversation, load_from_cache_file=False)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=100)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

total_samples = len(tokenized_dataset)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting training")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)