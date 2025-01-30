from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import json

# Load base LLaMA 3 model
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # Disable automatic device mapping
    load_in_4bit=False,  # Disables bitsandbytes (required for macOS)
    torch_dtype="auto"
)

# Move model to CPU
model = model.to("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply QLoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Load your JSONL dataset
def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("./Ollama/Instruction Bot/dataset/train.jsonl")
val_data = load_jsonl("./Ollama/Instruction Bot/dataset/val.jsonl")

# Convert dataset to proper format
def tokenize_function(example):
    return tokenizer(f"Question: {example['question']} Answer: {example['answer']}", truncation=True)

train_dataset = [tokenize_function(ex) for ex in train_data]
val_dataset = [tokenize_function(ex) for ex in val_data]

# Ensure data is on CPU
train_dataset = [{k: torch.tensor(v).to('cpu') for k, v in ex.items()} for ex in train_dataset]
val_dataset = [{k: torch.tensor(v).to('cpu') for k, v in ex.items()} for ex in val_dataset]

# Training arguments
training_args = TrainingArguments(
    output_dir="./Ollama/Instruction Bot/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduce batch size to lower memory usage
    per_device_eval_batch_size=1,   # Reduce batch size to lower memory usage
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save the trained model and tokenizer
output_dir = "./Ollama/Instruction Bot/Fine-Tuned-LLaMA3"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)