from .abstract_model import tp_model
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from typing import List
import torch
import os
import copy
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.rs_agent_prompt import RS_AGENT_PROMPT

from ..utils.log_utils import get_logger
inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("MistralLM_model",)

class MistralLM(tp_model):
    def __init__(
      self,  
      pretrained : str = None,
    ):
        self.model_path = pretrained
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="bfloat16",
            device_map=None,
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
        self.processor.padding_side = 'left'
        self.system_prompt = RS_AGENT_PROMPT
        self.max_new_tokens = None
        self.default_max_new_tokens = 256
        self.max_model_len = 8192
        logger.info(
            f"Initialized MistralLM with model={pretrained}, "
            f"max_model_len={self.max_model_len}, "
        )

    def generate_conversation_fn(self, text, role="user"):
        return [{"role": role, "content": text}]

    
    def append_conversation_fn(self, conversation, text, role):
        conversation.append({"role": role, "content": text})
        return conversation

    def prepend_system_prompt(self, conversation):
        """Ensure a system prompt is always the first message."""
        if not conversation or conversation[0].get("role") != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation
    
    def getitem_fn(self, meta_data, idx):
        res = meta_data[idx]
        return res
    
    def prepare_inputs(self, messages):
        SAFETY_MARGIN = 64
        max_model_len = self.max_model_len
        max_new_tokens = self.max_new_tokens or self.default_max_new_tokens
        limit = max_model_len - max_new_tokens - SAFETY_MARGIN
        if limit < 256: 
            raise ValueError("Wrong Limit")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text, add_special_tokens=False,return_tensors="pt")
        input_ids = encoded.input_ids[0] 
        attention_mask = encoded.attention_mask[0] 
        msgs = copy.deepcopy(messages)
        while input_ids.shape[0] > limit and len(msgs) > 4:
            msgs.pop(2)  # drop oldest (keep system, question)
            msgs.pop(2) # remove turn
            suffix = "\n<previous context truncated>"
            if suffix not in msgs[1]["content"]:
                msgs[1]["content"] += suffix
            logger.warning(f"Prompt too long ({len(input_ids)}). Truncating to {limit}.")
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            encoded = self.processor(text, add_special_tokens=False,return_tensors="pt")
            input_ids = encoded.input_ids[0] 
            attention_mask = encoded.attention_mask[0]  
        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
    
    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        if len(batch) == 0:
            return None
        input_ids_list = []
        attention_mask_list = []
        for item in batch:
            conv = self.prepend_system_prompt(item.conversation)
            enc = self.prepare_inputs(conv)
            input_ids_list.append(enc["input_ids"])
            attention_mask_list.append(enc["attention_mask"])

        inputs = self.processor.pad({
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list
            }, padding=True, return_tensors="pt"
        )
        return inputs
    
    def form_input_from_msg(self, messages):
        conv = self.prepend_system_prompt(messages)
        enc = self.prepare_inputs(conv)
        inputs = self.processor.pad(
            {
                "input_ids": [enc["input_ids"]],
                "attention_mask": [enc["attention_mask"]],
            },
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
        self.max_new_tokens = self.max_new_tokens if self.max_new_tokens is not None else self.default_max_new_tokens
        is_dynamic_batch = all(isinstance(x, DynamicBatchItem) for x in batch)
        
        if is_dynamic_batch:
            inputs = self.form_input_from_dynamic_batch(batch)
        else:
            inputs = self.form_input_from_msg(batch)

        device = next(self.model.parameters()).device
        inputs = BatchEncoding({k: v.to(device) for k, v in inputs.items()})

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if is_dynamic_batch:
            for item, output_text in zip(batch, output_texts):
                item.model_response.append(output_text)
                self.append_conversation_fn(
                    item.conversation, output_text, "assistant"
                )

        return output_texts

    
            
        
        
