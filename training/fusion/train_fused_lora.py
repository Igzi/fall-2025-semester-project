import json
import os
import sys
import math
import csv
import random
from transformers import LlamaForCausalLM, LlamaTokenizer
from dataclasses import dataclass
from typing import Dict, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.get_fused_peft_model import get_fused_peft_model

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def read_dataset(path: str) -> Dataset:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        ds = load_dataset("json", data_files=path, split="train")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = Dataset.from_list(data)
    else:
        raise ValueError("data_path must be .jsonl or .json")
    return ds


def train_val_split(ds: Dataset, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return ds, None
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_n = max(1, int(n * val_ratio))
    val_idxs = idxs[:val_n]
    train_idxs = idxs[val_n:]
    return ds.select(train_idxs), ds.select(val_idxs)


# --- NEW: Alpaca-aware collator with auto schema detection ---
from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers import AutoTokenizer

@dataclass
class AlpacaCollator:
    tokenizer: AutoTokenizer
    max_length: int
    prompt_input: str
    prompt_no_input: str

    def _extract_fields(self, ex: Dict) -> Dict[str, str]:
        """
        Supports two schemas:
          1) Alpaca:  instruction, input, output
          2) Legacy:  inputs, targets  -> mapped to (instruction=inputs, input="", output=targets)
        """
        if all(k in ex for k in ("instruction", "output")):
            instruction = ex["instruction"]
            input_txt = ex.get("input", "") or ""
            output = ex["output"]
        elif "inputs" in ex and "targets" in ex:
            instruction = ex["inputs"]
            input_txt = ""  # no separate input provided
            output = ex["targets"]
        else:
            raise ValueError(
                "Example must contain either ('instruction','output'[, 'input']) "
                "or ('inputs','targets'). Problem example keys: "
                f"{list(ex.keys())}"
            )
        # Ensure strings
        return {
            "instruction": str(instruction),
            "input": str(input_txt),
            "output": str(output),
        }

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = []
        prompt_lengths = []

        # Build full text: prompt + output; also build prompt-only to mask labels
        for ex in features:
            f = self._extract_fields(ex)
            has_input = len(f["input"].strip()) > 0
            prompt = (self.prompt_input if has_input else self.prompt_no_input).format(
                instruction=f["instruction"], input=f["input"]
            )
            full_text = prompt + f["output"]

            # For loss masking we need tokenized prompt length (no truncation)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            texts.append(full_text)
            prompt_lengths.append(len(prompt_ids))
        
        batch = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        labels = input_ids.clone()
        # Mask the prompt part so loss applies only to the response (output)
        for i, plen in enumerate(prompt_lengths):
            eff_len = min(plen, int(attn[i].sum().item()))
            labels[i, :eff_len] = -100  # ignore index

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def main():
    # Load config
    cfg = load_config("./training/fusion/config.json")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_csv_path = os.path.join(cfg["output_dir"], cfg["logging_csv"])

    set_seed(cfg["seed"])

    # Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        cfg['model_name'], torch_dtype=torch.float16
    )
    model.bfloat16()

    # LoRA rank=8
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj'],
    )

    # Dataset
    ds = read_dataset(cfg["data_path"])
    train_ds, val_ds = train_val_split(ds, cfg["val_ratio"], cfg["seed"])

    model = get_fused_peft_model(base_model=model, train_data=train_ds, exclude_list=None, lora_cfg=lora_cfg)

    model.enable_input_require_grads()

    collator = AlpacaCollator(
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        prompt_input=cfg["prompt_input"],
        prompt_no_input=cfg["prompt_no_input"],
    )

    # Training args
    eval_strategy = "epoch" if cfg["eval_steps"] is None else "steps"
    save_strategy = "epoch" if cfg["save_steps"] is None else "steps"

    train_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        evaluation_strategy=eval_strategy if val_ds is not None else "no",
        eval_steps=cfg["eval_steps"],
        save_strategy=save_strategy,
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        bf16=cfg["bf16"],
        fp16=cfg["fp16"] and not cfg["bf16"],
        report_to=[],
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,  # Add this line
    )

    # CSV logger callback
    class CSVLogger:
        def __init__(self, path):
            self.path = path
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "epoch", "loss", "eval_loss"])

        def log(self, state, logs: Dict):
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    int(state.global_step),
                    float(state.epoch) if state.epoch is not None else None,
                    logs.get("loss"),
                    logs.get("eval_loss"),
                ])

    csv_logger = CSVLogger(log_csv_path)

    from transformers import TrainerCallback, TrainerState, TrainerControl

    class WriteCSVCallback(TrainerCallback):
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs:
                csv_logger.log(state, logs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[WriteCSVCallback()],
    )

    print("\n" + "="*50)
    print("EVALUATING MODEL BEFORE TRAINING")
    print("="*50)
    
    if val_ds is not None:
        # Perform initial evaluation
        initial_metrics = trainer.evaluate()
        print(f"Initial evaluation metrics:")
        for key, value in initial_metrics.items():
            print(f"  {key}: {value}")
        
        # Log initial metrics to CSV
        csv_logger.log(trainer.state, initial_metrics)
        
        # Calculate initial perplexity
        if "eval_loss" in initial_metrics:
            initial_ppl = math.exp(initial_metrics["eval_loss"])
            print(f"  Initial Perplexity: {initial_ppl:.4f}")
    else:
        print("No validation dataset provided - skipping initial evaluation")
    
    trainer.train()

    # Save adapter
    adapter_dir = os.path.join(cfg["output_dir"], "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(cfg["output_dir"])

    # Final eval
    summary = {}
    if val_ds is not None:
        metrics = trainer.evaluate()
        summary["final_eval_loss"] = metrics.get("eval_loss")
        summary["final_eval_ppl"] = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else None
    summary["note"] = f"Losses logged in {log_csv_path}"
    with open(os.path.join(cfg["output_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()