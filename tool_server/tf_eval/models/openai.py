from .abstract_model import tp_model
import uuid
import os
from typing import List
import time
import copy
from .template_instruct import *
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.rs_agent_prompt import RS_AGENT_PROMPT
from ..utils.log_utils import get_logger

from openai import OpenAI
import tiktoken

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("openai_models")

class OpenAIModels(tp_model):
    def __init__(
        self,
        pretrained: str = "gpt-4o-mini", 
        api_key: str = None,
        max_model_len: int = None,
        temperature: float = 0.4,
        retries: int = 3
    ):
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided or missing in environment variable OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)
        self.model_name = pretrained
        self.temperature = temperature
        self.system_prompt = RS_AGENT_PROMPT
        self.retries = retries
        self.max_new_tokens = None
        self.default_max_new_tokens = 256
        if max_model_len is None:
            context_window = 8192
            if "32k" in pretrained:
                context_window = 32768
            elif "16k" in pretrained:
                context_window = 16384
            elif "gpt-4" in pretrained:
                context_window = 8192
            self.max_model_len = context_window
        else:
            self.max_model_len = max_model_len

        logger.info(
            f"[Init] OpenAIModels initialized with model='{self.model_name}', "
            f"temperature={self.temperature}, max_model_len={self.max_model_len}, retries={self.retries}"
        )
    # --- Conversation helpers ---

    def generate_conversation_fn(self, text, role="user"):
        return [{"role": role, "content": text}]

    def append_conversation_fn(self, conversation, text, role):
        conversation.append({"role": role, "content": text})
        return conversation

    def prepend_system_prompt(self, conversation):
        if not conversation or conversation[0].get("role") != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation

    def getitem_fn(self, meta_data, idx):
        return meta_data[idx]
    
    def truncate_conversation(self, messages):
        SAFETY_MARGIN = 64
        max_model_len = self.max_model_len
        max_new_tokens = self.max_new_tokens or self.default_max_new_tokens
        limit = max_model_len - max_new_tokens - 100 # Hold out 100 tokens due to potential errors in tiktoken calculation
        if limit < 256: 
            raise ValueError("Wrong Limit")
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        prompt_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_text += f"{role}: {content}\n"

        tokens = enc.encode(prompt_text)
        if len(tokens) <= limit:
            return messages

        logger.warning(f"Prompt too long ({len(tokens)} tokens). Truncating to {limit} tokens.")
        truncated_messages = messages.copy()

        while len(tokens) > limit and len(truncated_messages) > 4:
            truncated_messages.pop(2)  # drop oldest (keep system, question)
            truncated_messages.pop(2) # remove turn
            suffix = "\n<previous context truncated>"
            if suffix not in truncated_messages[1]["content"]:
                truncated_messages[1]["content"] += suffix
            # Recalculate token length
            prompt_text = ""
            for msg in truncated_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n"
            tokens = enc.encode(prompt_text)

        return truncated_messages

    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        if len(batch) == 0:
            return None
        messages = []
        for item in batch:
            conv = self.prepend_system_prompt(item.conversation)
            truncated_conv = self.truncate_conversation(conv)
            messages.append(truncated_conv)
        return messages

    def form_input_from_msg(self, messages):
        conv = self.prepend_system_prompt(messages)
        return self.truncate_conversation(conv)

    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
        self.max_new_tokens = self.max_new_tokens if self.max_new_tokens is not None else self.default_max_new_tokens
        is_dynamic_batch = all(isinstance(x, DynamicBatchItem) for x in batch)

        if is_dynamic_batch:
            inputs = self.form_input_from_dynamic_batch(batch)
        else:
            inputs = [self.form_input_from_msg(batch)]

        outputs = []
        for item, conv in zip(batch, inputs):
            retries = self.retries
            while retries > 0:
                try:
                    if "o4" in self.model_name:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=conv,
                            max_completion_tokens=self.max_new_tokens,
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=conv,
                            max_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                        )
                    print(response)                  
                    output_text = response.choices[0].message.content.strip()

                    if is_dynamic_batch:
                        # update dynamic batch items safely
                        item.model_response.append(output_text)
                        self.append_conversation_fn(
                            item.conversation, output_text, "assistant"
                        )

                    outputs.append(output_text)
                    break

                except Exception as e:
                    retries -= 1
                    logger.error(f"[OpenAI Error] {e} | Retries left: {retries}")
                    time.sleep(1)
                    if retries == 0:
                        outputs.append(f"Error generating response: {e}")

                # Return last output for step inference
        return [outputs[-1]] if outputs else None

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self
