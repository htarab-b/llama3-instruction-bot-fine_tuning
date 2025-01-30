import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,         # Efficient 4-bit quantization
    torch_dtype=torch.float16, # Uses float16 for better memory efficiency
    device_map="auto",         # Auto-detects CPU/MPS on Mac M2
)

FastLanguageModel.for_causal_lm(model)  # Enable for text generation

dataset = load_dataset("json", data_files={"train": "./Ollama/Instruction Bot/dataset/train.jsonl", "val": "./Ollama/Instruction Bot/dataset/val.jsonl"})

# Format dataset for LLaMA 3
def format_data(example):
    return {"text": f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"}

dataset = dataset.map(format_data)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

lora_config = LoraConfig(
    r=16,                          # Rank for LoRA
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Regularization dropout
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to key model layers
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./Ollama/Instruction Bot/fine-tuned-llama3",  
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  
    num_train_epochs=3,  
    logging_steps=10,  
    save_strategy="epoch",  
    push_to_hub=False,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./Ollama/Instruction Bot/llama3-finetuned")
tokenizer.save_pretrained("./Ollama/Instruction Bot/llama3-finetuned")

print("Fine-tuning complete! Model saved to './Ollama/Instruction Bot/llama3-finetuned'")