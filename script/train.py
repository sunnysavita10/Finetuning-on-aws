import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--train_data", type=str, default="/opt/ml/input/data/train/")
    return parser.parse_args()

def format_example(example):
    prompt = f"""### Instruction:
{example['instruction']}
### Input:
{example['input']}
### Response:
{example['output']}"""
    return {"text": prompt}


def tokenize_fn(example, tokenizer):
    tokens = tokenizer(
        example["text"], 
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():

    args = parse_args()

    print("\n===== Loading Dataset =====")
    dataset_path = os.path.join(args.train_data, "pharma_instruction_data.csv")
    dataset = load_dataset("csv", data_files={"train": dataset_path})["train"]

    dataset = dataset.map(format_example)

    print("\n===== Loading Tokenizer =====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_ds = dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    print("\n===== Loading Base Model in 4bit =====")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        load_in_4bit=True
    )

    print("\n===== Applying QLoRA =====")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    print("\n===== Setting Training Params =====")
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none"
    )

    print("\n===== Starting Training =====")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds
    )

    trainer.train()

    print("\n===== Saving Model =====")
    model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

    print("\n===== Training Completed Successfully =====")


if __name__ == "__main__":
    main()
