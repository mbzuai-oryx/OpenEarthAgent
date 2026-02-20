import unsloth 
from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
import logging
import os
import datasets
import torch
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT

# -------------------------
# Arguments
# -------------------------
@dataclass
class ScriptArguments:
    data_files: str
    dataset_train_split: str = "train"


@dataclass
class ModelArguments:
    model_name_or_path: str
    load_in_4bit: bool = False

# -------------------------
# Logging
# -------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)


# -------------------------
# Dataset conversion
# -------------------------
def convert_example(example):
    """
    Converts dataset format 
    """
    conversations = []

    # system message
    conversations.append({
        "role": "system",
        "content": RS_AGENT_PROMPT,
    })

    for item in example.get("conversation", []):
        role = "user" if item["from"] == "human" else "assistant"
        content = item["value"].replace("<AGENT_PROMPT>\n\nQuestion: ", "")
        conversations.append({
            "role": role,
            "content": content,
        })

    return {"conversations": conversations}


# -------------------------
# Main
# -------------------------
def main(script_args, model_args, training_args):
    set_seed(training_args.seed)
    logger = setup_logging()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(f"Training args: {training_args}")

    # -------------------------
    # Load dataset
    # -------------------------
    raw_dataset = load_dataset(
        "json",
        data_files=script_args.data_files,
        split=script_args.dataset_train_split,
    )

    dataset = raw_dataset.map(
        convert_example,
        remove_columns=raw_dataset.column_names,
    )

    # Standardize to Unsloth / ShareGPT format
    dataset = standardize_data_formats(dataset)

    # -------------------------
    # Load model + tokenizer (Unsloth)
    # -------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,
        device_map={"": local_rank},
    )

    # Apply Qwen-style chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )

    # -------------------------
    # Format dataset text
    # -------------------------
    def formatting_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(
        formatting_func,
        batched=True,
    )

    # -------------------------
    # Trainer
    # -------------------------
    training_args.dataset_text_field = "text"
    training_args.report_to = ["tensorboard"]
    training_args.do_train = True
    training_args.dataset_num_proc = 1
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
    )

    # -------------------------
    # Mask user turns (KEY STEP)
    # -------------------------
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    # -------------------------
    # Train
    # -------------------------
    logger.info("*** Train ***")
    trainer.train()

    # -------------------------
    # Save
    # -------------------------
    logger.info("*** Save model ***")
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info(f"Model saved to {training_args.output_dir}")


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(
        (ScriptArguments, ModelArguments, SFTConfig)
    )
    script_args, model_args, training_args = parser.parse_args_into_dataclasses()
    main(script_args, model_args, training_args)
